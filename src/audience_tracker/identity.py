"""Global Identity Manager — the source of truth for audience identities.

Implements the Identity Rules from the spec:

  Rule 1  Every newly observed audience member receives a unique Global ID (GID).
  Rule 2  A new track matching a previous identity above the similarity
          threshold reuses that identity's GID.
  Rule 3  Otherwise a new GID is created.
  Rule 4  GIDs are never reused during a session (monotonic counter).
  Rule 5  Tracker IDs are internal and may change; GIDs remain stable. External
          consumers only ever see GIDs.

Thread-safe: the pipeline thread calls :meth:`update` while API threads read
via :meth:`active`, :meth:`get`, :meth:`stats`, :meth:`snapshot`.
"""

from __future__ import annotations

import threading
import time
from typing import Optional

from .config import IdentityConfig, ReIDConfig
from .models import AudienceState, Embedding, Identity, Track
from .vecmath import cosine_similarity, ema, normalize


class IdentityManager:
    def __init__(self, identity_cfg: IdentityConfig, reid_cfg: ReIDConfig) -> None:
        self._cfg = identity_cfg
        self._reid = reid_cfg
        self._identities: dict[int, Identity] = {}
        self._track_to_gid: dict[int, int] = {}
        self._next_gid = 1
        self._total_created = 0
        self._id_switches = 0
        self._lock = threading.RLock()

    # ------------------------------------------------------------------ #
    # Core update (called once per processed frame by the pipeline)
    # ------------------------------------------------------------------ #
    def update(
        self,
        tracks: list[Track],
        embeddings: Optional[dict[int, Embedding]] = None,
        now: Optional[float] = None,
    ) -> list[AudienceState]:
        """Reconcile tracker output with global identities.

        ``embeddings`` maps ``track_id`` -> appearance embedding for the tracks
        that were re-embedded this frame (may be partial or empty).
        Returns the currently-visible audience states.
        """
        now = time.time() if now is None else now
        embeddings = embeddings or {}

        with self._lock:
            present_ids = {t.track_id for t in tracks}

            # 1. Update tracks already bound to an identity; queue the rest.
            # Bindings survive short detection misses: the tracker keeps a lost
            # track alive (lost_track_buffer) and re-emits the SAME id when the
            # person is re-detected, and that continuity must map back to the
            # same GID — even with ReID disabled.
            unbound: list[Track] = []
            for t in tracks:
                gid = self._track_to_gid.get(t.track_id)
                ident = self._identities.get(gid) if gid is not None else None
                if ident is not None and ident.active_track_id in (None, t.track_id):
                    emb = embeddings.get(t.track_id)
                    if (
                        not ident.visible
                        and emb
                        and ident.embedding_avg is not None
                        and self._reid.enabled
                        and cosine_similarity(emb, ident.embedding_avg)
                        < self._reid.rebind_veto_threshold
                    ):
                        # The tracker re-emitted this id after a miss but the
                        # appearance clearly disagrees — it re-associated onto
                        # a different person. Break the binding and let step 2
                        # arbitrate by appearance (Rule 2/3) instead.
                        self._track_to_gid.pop(t.track_id, None)
                        unbound.append(t)
                        continue
                    self._touch(ident, t, emb, now)
                else:
                    if gid is not None:
                        # Stale binding: the identity was forgotten, or already
                        # claimed by another live track via ReID recovery.
                        self._track_to_gid.pop(t.track_id, None)
                    unbound.append(t)

            # 2. Resolve new/unbound tracks: ReID recovery (Rule 2) else new GID.
            for t in unbound:
                emb = embeddings.get(t.track_id)
                gid = self._match_lost(emb, now) if (emb and self._reid.enabled) else None
                if gid is not None:
                    ident = self._identities[gid]
                    ident.recoveries += 1
                    # The tracker issued a new id for this person (otherwise
                    # step 1 would have rebound it): a recovered id switch.
                    self._id_switches += 1
                    ident.active_track_id = t.track_id
                    self._unbind_gid(gid, keep=t.track_id)
                    self._track_to_gid[t.track_id] = gid
                    self._touch(ident, t, emb, now, recovered=True)
                else:
                    self._create(t, emb, now)

            # 3. Identities whose track vanished this frame become "lost". The
            # track binding is intentionally kept: a short miss makes the
            # tracker re-emit the same id, which must resume the same GID.
            for ident in self._identities.values():
                if ident.active_track_id is not None and ident.active_track_id not in present_ids:
                    ident.active_track_id = None
                    ident.visible = False

            # 4. Forget long-gone identities (bounded memory). GID never reused.
            self._gc(now)

            return [i.to_state() for i in self._identities.values() if i.visible]

    # ------------------------------------------------------------------ #
    # Read API (thread-safe)
    # ------------------------------------------------------------------ #
    def active(self, now: Optional[float] = None) -> list[AudienceState]:
        now = time.time() if now is None else now
        with self._lock:
            return [self._state(i, now) for i in self._identities.values() if i.visible]

    def snapshot(self, now: Optional[float] = None) -> list[AudienceState]:
        """All identities still held in memory (visible + recently lost)."""
        now = time.time() if now is None else now
        with self._lock:
            return [self._state(i, now) for i in self._identities.values()]

    def get(self, gid: int, now: Optional[float] = None) -> Optional[AudienceState]:
        now = time.time() if now is None else now
        with self._lock:
            ident = self._identities.get(gid)
            return self._state(ident, now) if ident else None

    def stats(self) -> dict:
        with self._lock:
            active = sum(1 for i in self._identities.values() if i.visible)
            return {"active_people": active, "total_people_seen": self._total_created}

    def known_track_ids(self) -> set[int]:
        """Track IDs currently bound to an identity (used for ReID scheduling)."""
        with self._lock:
            return set(self._track_to_gid)

    def returning_track_ids(self) -> set[int]:
        """Track IDs bound to a currently-invisible identity.

        These are about to resume a GID if the tracker re-emits them; the
        pipeline embeds them immediately so the rebind is appearance-checked
        (see the rebind veto in :meth:`update`)."""
        with self._lock:
            return {
                tid
                for tid, gid in self._track_to_gid.items()
                if gid in self._identities and not self._identities[gid].visible
            }

    def counters(self) -> dict:
        with self._lock:
            # visible <=> bound to a live track (set/cleared together), so the
            # two counts coincide; both keys stay for the /metrics contract.
            active = sum(1 for i in self._identities.values() if i.visible)
            return {
                "active_people": active,
                "active_tracks": active,
                "total_people_seen": self._total_created,
                "id_switches": self._id_switches,
                "recoveries": sum(i.recoveries for i in self._identities.values()),
            }

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _create(self, t: Track, emb: Optional[Embedding], now: float) -> int:
        gid = self._next_gid
        self._next_gid += 1
        self._total_created += 1
        ident = Identity(
            gid=gid,
            first_seen=now,
            last_seen=now,
            embedding_avg=normalize(emb) if emb else None,
            visible=True,
            active_track_id=t.track_id,
            last_bbox=t.bbox,
            last_confidence=t.confidence,
            update_count=1,
        )
        self._identities[gid] = ident
        self._track_to_gid[t.track_id] = gid
        return gid

    def _touch(
        self,
        ident: Identity,
        t: Track,
        emb: Optional[Embedding],
        now: float,
        recovered: bool = False,
    ) -> None:
        # Accumulate visible time only across consecutive visible frames.
        if ident.visible and not recovered:
            ident.duration_seen_seconds += max(0.0, now - ident.last_seen)
        ident.visible = True
        ident.last_seen = now
        ident.last_bbox = t.bbox
        ident.last_confidence = t.confidence
        ident.active_track_id = t.track_id
        ident.update_count += 1
        if emb:
            ident.embedding_avg = ema(ident.embedding_avg, emb, self._reid.embedding_alpha)

    def _match_lost(self, emb: Embedding, now: float) -> Optional[int]:
        """Best lost identity above the similarity threshold, else None."""
        best_gid: Optional[int] = None
        best_sim = self._reid.similarity_threshold
        for gid, ident in self._identities.items():
            if ident.visible or ident.active_track_id is not None or ident.embedding_avg is None:
                continue
            if now - ident.last_seen > self._cfg.lost_timeout_seconds:
                continue
            sim = cosine_similarity(emb, ident.embedding_avg)
            if sim >= best_sim:
                best_sim = sim
                best_gid = gid
        return best_gid

    def _gc(self, now: float) -> None:
        stale = [
            gid
            for gid, i in self._identities.items()
            if not i.visible and (now - i.last_seen) > self._cfg.forget_timeout_seconds
        ]
        for gid in stale:
            del self._identities[gid]
            self._unbind_gid(gid)

    def _unbind_gid(self, gid: int, keep: Optional[int] = None) -> None:
        """Drop track bindings pointing at ``gid`` (except ``keep``)."""
        for tid in [t for t, g in self._track_to_gid.items() if g == gid and t != keep]:
            del self._track_to_gid[tid]

    @staticmethod
    def _state(ident: Identity, now: float) -> AudienceState:
        state = ident.to_state()
        if ident.visible:
            # Live extension of duration for display only (not persisted).
            state.duration_seen_seconds = round(
                ident.duration_seen_seconds + max(0.0, now - ident.last_seen), 3
            )
        return state
