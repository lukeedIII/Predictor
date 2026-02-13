# ============================================================
#  Nexus Shadow-Quant — Embed Python Script
# ============================================================
#  Downloads Python 3.11 Embeddable and installs all dependencies.
#  Result: desktop/python_embedded/ (~1.5 GB with PyTorch CUDA)
#
#  Usage: .\build_scripts\embed_python.ps1
# ============================================================

param(
    [string]$OutputDir = (Join-Path (Join-Path (Join-Path $PSScriptRoot "..") "desktop") "python_embedded"),
    [string]$PythonVersion = "3.11.9",
    [switch]$Force
)

$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "═══════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "  NEXUS SHADOW-QUANT — Python Embedding Script" -ForegroundColor Cyan
Write-Host "═══════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$RequirementsFile = Join-Path $ProjectRoot "requirements.txt"
$PythonZipUrl = "https://www.python.org/ftp/python/$PythonVersion/python-$PythonVersion-embed-amd64.zip"
$PythonZipFile = Join-Path $env:TEMP "python-$PythonVersion-embed-amd64.zip"
$GetPipUrl = "https://bootstrap.pypa.io/get-pip.py"
$GetPipFile = Join-Path $env:TEMP "get-pip.py"

# ── Step 0: Cleanup ─────────────────────────────────
if (Test-Path $OutputDir) {
    if ($Force) {
        Write-Host "  Cleaning existing python_embedded/..." -ForegroundColor Yellow
        Remove-Item $OutputDir -Recurse -Force
    } else {
        Write-Host "  python_embedded/ already exists. Use -Force to rebuild." -ForegroundColor Yellow
        Write-Host "  Skipping Python download, installing packages only." -ForegroundColor Yellow
    }
}

# ── Step 1: Download Python Embeddable ───────────────
if (!(Test-Path $OutputDir)) {
    Write-Host ""
    Write-Host "  Step 1/6: Downloading Python $PythonVersion Embeddable..." -ForegroundColor Green

    if (!(Test-Path $PythonZipFile)) {
        Invoke-WebRequest -Uri $PythonZipUrl -OutFile $PythonZipFile -UseBasicParsing
    }

    Write-Host "  Extracting to $OutputDir..."
    New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null
    Expand-Archive -Path $PythonZipFile -DestinationPath $OutputDir -Force

    Write-Host "  ✅ Python $PythonVersion extracted" -ForegroundColor Green
}

# ── Step 2: Enable pip in embeddable Python ──────────
Write-Host ""
Write-Host "  Step 2/6: Enabling pip..." -ForegroundColor Green

$PthFile = Get-ChildItem "$OutputDir\python*._pth" | Select-Object -First 1
if ($PthFile) {
    $content = Get-Content $PthFile.FullName
    if ($content -match "^#import site") {
        $content = $content -replace "^#import site", "import site"
        Set-Content -Path $PthFile.FullName -Value $content
        Write-Host "  Uncommented 'import site' in $($PthFile.Name)"
    }
    # Add Lib\site-packages to path
    if ($content -notmatch "Lib\\site-packages") {
        Add-Content -Path $PthFile.FullName -Value "Lib\site-packages"
        Write-Host "  Added Lib\site-packages to path"
    }
}

# ── Step 3: Install pip ─────────────────────────────
Write-Host ""
Write-Host "  Step 3/6: Installing pip..." -ForegroundColor Green

$EmbeddedPython = Join-Path $OutputDir "python.exe"

if (!(Test-Path $GetPipFile)) {
    Invoke-WebRequest -Uri $GetPipUrl -OutFile $GetPipFile -UseBasicParsing
}

# Create Lib\site-packages
$SitePackages = Join-Path (Join-Path $OutputDir "Lib") "site-packages"
New-Item -ItemType Directory -Path $SitePackages -Force | Out-Null

& $EmbeddedPython $GetPipFile --no-warn-script-location 2>&1 | ForEach-Object { Write-Host "    $_" }
Write-Host "  ✅ pip installed" -ForegroundColor Green

# ── Step 4: Install PyTorch with CUDA ────────────────
Write-Host ""
Write-Host "  Step 4/6: Installing PyTorch with CUDA 12.1 (~1.2 GB)..." -ForegroundColor Green
Write-Host "  This will take several minutes..." -ForegroundColor Yellow

& $EmbeddedPython -m pip install torch --index-url https://download.pytorch.org/whl/cu121 `
    --no-warn-script-location --target $SitePackages 2>&1 | ForEach-Object {
    if ($_ -match "Downloading|Installing|Successfully") { Write-Host "    $_" }
}
Write-Host "  ✅ PyTorch with CUDA installed" -ForegroundColor Green

# ── Step 5: Install remaining requirements ───────────
Write-Host ""
Write-Host "  Step 5/6: Installing remaining packages..." -ForegroundColor Green

& $EmbeddedPython -m pip install -r $RequirementsFile `
    --no-warn-script-location --target $SitePackages `
    --ignore-installed torch 2>&1 | ForEach-Object {
    if ($_ -match "Downloading|Installing|Successfully|Requirement") { Write-Host "    $_" }
}
Write-Host "  ✅ All packages installed" -ForegroundColor Green

# ── Step 6: Verify ───────────────────────────────────
Write-Host ""
Write-Host "  Step 6/6: Verifying installation..." -ForegroundColor Green

$VerifyScript = @"
import sys
print(f'Python: {sys.version}')
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
import xgboost
print(f'XGBoost: {xgboost.__version__}')
import pandas
print(f'Pandas: {pandas.__version__}')
import fastapi
print(f'FastAPI: {fastapi.__version__}')
print('ALL OK')
"@

$VerifyFile = Join-Path $env:TEMP "verify_embed.py"
Set-Content -Path $VerifyFile -Value $VerifyScript
& $EmbeddedPython $VerifyFile 2>&1 | ForEach-Object { Write-Host "    $_" }

# ── Report ────────────────────────────────────────────
$Size = (Get-ChildItem $OutputDir -Recurse | Measure-Object Length -Sum).Sum / 1GB
Write-Host ""
Write-Host "═══════════════════════════════════════════════════════" -ForegroundColor Green
Write-Host "  ✅ Python embedded successfully!" -ForegroundColor Green
Write-Host "  📁 Location: $OutputDir" -ForegroundColor Green
Write-Host "  📦 Size: $([math]::Round($Size, 2)) GB" -ForegroundColor Green
Write-Host "═══════════════════════════════════════════════════════" -ForegroundColor Green
Write-Host ""
