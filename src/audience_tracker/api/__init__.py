"""REST + WebSocket API service for the audience tracking system."""

from .app import create_app

__all__ = ["create_app"]
