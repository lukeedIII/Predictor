# ============================================================
#  Nexus Shadow-Quant — Master Build Script
# ============================================================
#  One command to create the distributable installer.
#
#  Usage: .\build_scripts\build_installer.ps1
#         .\build_scripts\build_installer.ps1 -SkipPython  (if python_embedded exists)
# ============================================================

param(
    [switch]$SkipPython,
    [switch]$Force
)

$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "═══════════════════════════════════════════════════════" -ForegroundColor Magenta
Write-Host "  NEXUS SHADOW-QUANT — MASTER BUILD SCRIPT" -ForegroundColor Magenta
Write-Host "═══════════════════════════════════════════════════════" -ForegroundColor Magenta
Write-Host ""

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$DesktopDir = Join-Path $ProjectRoot "desktop"
$PythonEmbedded = Join-Path $DesktopDir "python_embedded"
$PythonBackend = Join-Path $DesktopDir "python_backend"
$ReleaseDir = Join-Path $DesktopDir "release"

$StartTime = Get-Date

# ══════════════════════════════════════════════════════
#  Step 1: Embed Python Runtime
# ══════════════════════════════════════════════════════

if ($SkipPython -and (Test-Path $PythonEmbedded)) {
    Write-Host "  Step 1/6: Skipping Python embedding (already exists)" -ForegroundColor Yellow
} else {
    Write-Host "  Step 1/6: Embedding Python runtime..." -ForegroundColor Green
    & (Join-Path $PSScriptRoot "embed_python.ps1") -OutputDir $PythonEmbedded -Force:$Force
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  ❌ Python embedding failed!" -ForegroundColor Red
        exit 1
    }
}

# ══════════════════════════════════════════════════════
#  Step 2: Copy Python Backend Source
# ══════════════════════════════════════════════════════

Write-Host ""
Write-Host "  Step 2/6: Copying Python backend source..." -ForegroundColor Green

if (Test-Path $PythonBackend) {
    Remove-Item $PythonBackend -Recurse -Force
}
New-Item -ItemType Directory -Path $PythonBackend -Force | Out-Null

# Copy all Python source files
$PythonFiles = @(
    "api_server.py", "predictor.py", "config.py", "data_collector.py",
    "download_historical.py", "first_run.py", "system_check.py",
    "paper_trader.py", "sentiment_engine.py", "alt_data.py",
    "math_core.py", "quant_models.py", "nexus_logger.py",
    "hardware_profiler.py", "notifications.py", "whale_monitor.py",
    "backtester.py", "backtest_utils.py", "baselines.py",
    "run_backtest.py", "run_backtest_parallel.py",
    "twitter_scraper.py", "fintech_theme.py", "main.py", "app.py"
)

foreach ($file in $PythonFiles) {
    $src = Join-Path $ProjectRoot $file
    if (Test-Path $src) {
        Copy-Item $src $PythonBackend
        Write-Host "    ✓ $file" -ForegroundColor DarkGray
    } else {
        Write-Host "    ⚠ $file (not found, skipping)" -ForegroundColor Yellow
    }
}

# Copy requirements.txt
Copy-Item (Join-Path $ProjectRoot "requirements.txt") $PythonBackend

# Create empty data directories (will be populated at runtime in AppData)
foreach ($dir in @("data", "models", "logs")) {
    New-Item -ItemType Directory -Path (Join-Path $PythonBackend $dir) -Force | Out-Null
    # Create a .gitkeep so the dir isn't empty
    New-Item -ItemType File -Path (Join-Path (Join-Path $PythonBackend $dir) ".gitkeep") -Force | Out-Null
}

$BackendCount = (Get-ChildItem "$PythonBackend\*.py" | Measure-Object).Count
Write-Host "  ✅ $BackendCount Python files copied" -ForegroundColor Green

# ══════════════════════════════════════════════════════
#  Step 3: Build React Frontend
# ══════════════════════════════════════════════════════

Write-Host ""
Write-Host "  Step 3/6: Building React frontend..." -ForegroundColor Green

Push-Location $DesktopDir
try {
    npm run build 2>&1 | ForEach-Object { Write-Host "    $_" }
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  ❌ Frontend build failed!" -ForegroundColor Red
        exit 1
    }
} finally {
    Pop-Location
}

Write-Host "  ✅ Frontend built" -ForegroundColor Green

# ══════════════════════════════════════════════════════
#  Step 4: Verify Package Structure
# ══════════════════════════════════════════════════════

Write-Host ""
Write-Host "  Step 4/6: Verifying package structure..." -ForegroundColor Green

$RequiredPaths = @(
    (Join-Path (Join-Path $DesktopDir "dist") "index.html"),
    (Join-Path (Join-Path $DesktopDir "electron") "main.js"),
    (Join-Path (Join-Path $DesktopDir "electron") "preload.js"),
    (Join-Path (Join-Path $DesktopDir "electron") "splash.html"),
    (Join-Path $PythonEmbedded "python.exe"),
    (Join-Path $PythonBackend "api_server.py"),
    (Join-Path $PythonBackend "predictor.py"),
    (Join-Path $PythonBackend "config.py"),
    (Join-Path $PythonBackend "first_run.py"),
    (Join-Path $PythonBackend "system_check.py"),
    (Join-Path (Join-Path $DesktopDir "public") "DISCLAIMER.txt")
)

$MissingFiles = @()
foreach ($reqPath in $RequiredPaths) {
    if (Test-Path $reqPath) {
        Write-Host "    ✓ $(Split-Path $reqPath -Leaf)" -ForegroundColor DarkGray
    } else {
        Write-Host "    ✗ MISSING: $reqPath" -ForegroundColor Red
        $MissingFiles += $reqPath
    }
}

if ($MissingFiles.Count -gt 0) {
    Write-Host ""
    Write-Host "  ❌ $($MissingFiles.Count) required files missing! Cannot build installer." -ForegroundColor Red
    exit 1
}

Write-Host "  ✅ All required files present" -ForegroundColor Green

# ══════════════════════════════════════════════════════
#  Step 5: Build Electron Installer
# ══════════════════════════════════════════════════════

Write-Host ""
Write-Host "  Step 5/6: Building NSIS installer..." -ForegroundColor Green
Write-Host "  This may take 2-5 minutes..." -ForegroundColor Yellow

Push-Location $DesktopDir
try {
    npx electron-builder --win --x64 2>&1 | ForEach-Object { Write-Host "    $_" }
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  ❌ Installer build failed!" -ForegroundColor Red
        exit 1
    }
} finally {
    Pop-Location
}

Write-Host "  ✅ Installer built" -ForegroundColor Green

# ══════════════════════════════════════════════════════
#  Step 6: Report
# ══════════════════════════════════════════════════════

Write-Host ""
Write-Host "  Step 6/6: Final report..." -ForegroundColor Green

$Installer = Get-ChildItem "$ReleaseDir\*.exe" -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 1
$Duration = (Get-Date) - $StartTime

if ($Installer) {
    $InstallerSize = [math]::Round($Installer.Length / 1GB, 2)
    $PythonSize = [math]::Round((Get-ChildItem $PythonEmbedded -Recurse | Measure-Object Length -Sum).Sum / 1GB, 2)
    $BackendSize = [math]::Round((Get-ChildItem $PythonBackend -Recurse | Measure-Object Length -Sum).Sum / 1MB, 1)
    
    Write-Host ""
    Write-Host "═══════════════════════════════════════════════════════" -ForegroundColor Green
    Write-Host "  ✅ BUILD COMPLETE" -ForegroundColor Green
    Write-Host "═══════════════════════════════════════════════════════" -ForegroundColor Green
    Write-Host ""
    Write-Host "  📦 Installer:        $($Installer.Name)" -ForegroundColor White
    Write-Host "  📁 Location:         $($Installer.FullName)" -ForegroundColor White
    Write-Host "  📐 Installer Size:   $InstallerSize GB" -ForegroundColor White
    Write-Host "  🐍 Python Runtime:   $PythonSize GB" -ForegroundColor White
    Write-Host "  📜 Backend Source:    $BackendSize MB" -ForegroundColor White
    Write-Host "  ⏱  Build Time:       $([math]::Round($Duration.TotalMinutes, 1)) minutes" -ForegroundColor White
    Write-Host ""
    Write-Host "  Next: Double-click the .exe to install!" -ForegroundColor Cyan
    Write-Host ""
} else {
    Write-Host ""
    Write-Host "  ⚠ No installer found in $ReleaseDir" -ForegroundColor Yellow
    Write-Host "  Check the electron-builder output above for errors." -ForegroundColor Yellow
    Write-Host ""
}
