<#
.SYNOPSIS
  One-command Windows setup for the Audience Tracker (local / TouchDesigner path).

.DESCRIPTION
  Creates a virtual environment, installs the right PyTorch build for your GPU,
  installs the detection/tracking stack, and verifies the result with
  `audience-tracker doctor`. Designed to be re-run safely.

  Handles the common scenarios so you don't have to think about them:
    * finds a suitable Python (3.10-3.12) or tells you exactly what to install
    * detects your NVIDIA GPU + CUDA version and picks the matching torch wheel
    * falls back to a CPU build (with a clear warning) when there's no GPU
    * verifies torch actually sees the GPU at the end

.PARAMETER Reid
  Also install the optional ReID stack (torchreid / OSNet).

.PARAMETER Cpu
  Force a CPU-only PyTorch install (no CUDA). Much slower; for testing only.

.PARAMETER CudaTag
  Override the auto-detected CUDA wheel tag, e.g. "cu124", "cu121", "cu118".

.PARAMETER Python
  Path to a specific python.exe to build the venv from.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\install_windows.ps1
.EXAMPLE
  scripts\install_windows.bat -Reid
#>
[CmdletBinding()]
param(
    [switch]$Reid,
    [switch]$Cpu,
    [string]$CudaTag = "",
    [string]$Python = ""
)

$ErrorActionPreference = "Stop"

function Info($m)  { Write-Host "==> $m" -ForegroundColor Cyan }
function Good($m)  { Write-Host "[OK] $m" -ForegroundColor Green }
function Warn($m)  { Write-Host "[!!] $m" -ForegroundColor Yellow }
function Die($m)   { Write-Host "[XX] $m" -ForegroundColor Red; exit 1 }

$RepoRoot = (Resolve-Path "$PSScriptRoot\..").Path
$VenvDir  = Join-Path $RepoRoot ".venv"
$VenvPy   = Join-Path $VenvDir "Scripts\python.exe"

# --------------------------------------------------------------------------- #
# 1. Find a suitable Python (3.10 - 3.12; torch has no wheels for 3.13+ yet).
# --------------------------------------------------------------------------- #
# Returns a hashtable @{ Exe = <path>; Pre = @(<prefix args>) } for a usable
# Python 3.10-3.12, or $null. Pre holds e.g. @("-3.11") for the py launcher.
function Resolve-Python {
    $candidates = @()
    if ($Python) {
        if (-not (Test-Path $Python)) { Die "Specified Python not found: $Python" }
        $candidates += ,@{ Exe = $Python; Pre = @() }
    } else {
        foreach ($v in @("-3.12", "-3.11", "-3.10")) {
            $candidates += ,@{ Exe = "py"; Pre = @($v) }
        }
        $candidates += ,@{ Exe = "python"; Pre = @() }
    }
    foreach ($c in $candidates) {
        if (-not (Get-Command $c.Exe -ErrorAction SilentlyContinue)) { continue }
        try {
            $probe = $c.Pre + @("-c", "import sys;print('%d.%d'%sys.version_info[:2])")
            $ver = (& $c.Exe @probe 2>$null)
            if ($ver -match "^3\.(\d+)$") {
                $minor = [int]$Matches[1]
                if ($minor -ge 10 -and $minor -le 12) { return $c }
            }
        } catch { }
    }
    return $null
}

Info "Locating Python 3.10-3.12 ..."
$py = Resolve-Python
if (-not $py) {
    Die @"
No suitable Python found (need 3.10, 3.11 or 3.12).
Install one from https://www.python.org/downloads/  (tick 'Add to PATH'),
then re-run this script. Python 3.13+ is not yet supported by PyTorch wheels.
"@
}
Good ("Using Python: " + $py.Exe + " " + ($py.Pre -join " "))

# --------------------------------------------------------------------------- #
# 2. Create / reuse the virtual environment.
# --------------------------------------------------------------------------- #
if (Test-Path $VenvPy) {
    Good "Reusing existing venv at .venv"
} else {
    Info "Creating venv at .venv ..."
    & $py.Exe @($py.Pre + @("-m", "venv", $VenvDir))
    if (-not (Test-Path $VenvPy)) { Die "venv creation failed." }
    Good "venv created"
}

Info "Upgrading pip ..."
& $VenvPy -m pip install --upgrade pip --quiet

# --------------------------------------------------------------------------- #
# 3. Decide on the PyTorch build (CUDA vs CPU).
# --------------------------------------------------------------------------- #
function Detect-CudaTag {
    if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) { return $null }
    $smi = & nvidia-smi 2>$null | Out-String
    if ($smi -match "CUDA Version:\s*(\d+)\.(\d+)") {
        $maj = [int]$Matches[1]; $min = [int]$Matches[2]
        if     ($maj -gt 12 -or ($maj -eq 12 -and $min -ge 4)) { return "cu124" }
        elseif ($maj -eq 12)                                   { return "cu121" }
        elseif ($maj -eq 11 -and $min -ge 8)                   { return "cu118" }
        else                                                  { return "cu121" }
    }
    return "cu124"  # GPU present but couldn't parse version -> sane default
}

$torchIndex = ""
if ($Cpu) {
    Warn "Installing CPU-only PyTorch (forced with -Cpu). This will be slow."
} else {
    $tag = if ($CudaTag) { $CudaTag } else { Detect-CudaTag }
    if (-not $tag) {
        Warn "No NVIDIA GPU detected (nvidia-smi missing). Installing CPU-only PyTorch."
        Warn "If this machine has a GPU, install the driver, then re-run."
    } else {
        $torchIndex = "https://download.pytorch.org/whl/$tag"
        Good "GPU detected -> PyTorch build: $tag"
    }
}

Info "Installing PyTorch (torch + torchvision) ..."
if ($torchIndex) {
    & $VenvPy -m pip install torch torchvision --index-url $torchIndex
} else {
    & $VenvPy -m pip install torch torchvision
}
if ($LASTEXITCODE -ne 0) { Die "PyTorch install failed (see pip output above)." }

# --------------------------------------------------------------------------- #
# 4. Install the package extras.
# --------------------------------------------------------------------------- #
Push-Location $RepoRoot
try {
    $extra = if ($Reid) { ".[detect,reid]" } else { ".[detect]" }
    Info "Installing audience-tracker $extra ..."
    & $VenvPy -m pip install -e $extra
    if ($LASTEXITCODE -ne 0) {
        if ($Reid) {
            Warn "Install failed with [reid]. torchreid is awkward on Windows."
            Warn "Retry without -Reid (run with --no-reid), or install torchreid from source:"
            Warn "  .venv\Scripts\python -m pip install git+https://github.com/KaiyangZhou/deep-person-reid.git"
        }
        Die "Package install failed."
    }
} finally {
    Pop-Location
}

# --------------------------------------------------------------------------- #
# 5. Verify.
# --------------------------------------------------------------------------- #
Info "Verifying environment ..."
$requireCuda = if ($torchIndex) { "--require-cuda" } else { "" }
$doctorArgs = @("-m", "audience_tracker.cli", "doctor", "--require", "detect")
if ($requireCuda) { $doctorArgs += $requireCuda }
& $VenvPy @doctorArgs
$doctorExit = $LASTEXITCODE

Write-Host ""
if ($doctorExit -eq 0) {
    Good "Setup complete."
} else {
    Warn "Setup finished but verification reported problems (see above)."
}

Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "  1) Start the service (service owns the camera, ReID off):"
Write-Host "       scripts\run_windows.bat"
Write-Host "     or directly:"
Write-Host "       .venv\Scripts\audience-tracker serve --backend real --device cuda --source 0 --no-reid"
Write-Host "  2) Wire up TouchDesigner: see docs\WINDOWS_TOUCHDESIGNER.md"
Write-Host "  3) Re-check anytime:  .venv\Scripts\audience-tracker doctor"
exit $doctorExit
