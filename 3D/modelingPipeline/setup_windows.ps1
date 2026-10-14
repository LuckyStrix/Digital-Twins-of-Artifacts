<#
.SYNOPSIS
    One-shot, no-admin setup of the 3D reconstruction app on native Windows.

.DESCRIPTION
    Everything goes inside this folder (3D\modelingPipeline); nothing is
    installed system-wide and no admin rights are needed:

      venv\      Python virtualenv: CUDA build of torch + requirements.txt
      colmap\    the official COLMAP CUDA release for Windows (COLMAP.bat)
      tools\     exiftool.exe (+ exiftool_files\)

    Then it runs check_setup.py. Safe to re-run: finished steps are skipped.
    See ..\WINDOWS_SETUP.md.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File setup_windows.ps1
#>
param(
    # CUDA 12 builds of torch to try, in order. Must stay CUDA 12: the
    # onnxruntime-gpu pinned in requirements.txt (<1.27) is built for CUDA 12.
    [string[]]$TorchCuda = @("cu128", "cu126"),
    [switch]$SkipColmap,
    [switch]$SkipExiftool,
    # Re-download COLMAP / exiftool even if already present.
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"   # Invoke-WebRequest is very slow with the progress bar
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

$Here = $PSScriptRoot
$Venv = Join-Path $Here "venv"
$VenvPy = Join-Path $Venv "Scripts\python.exe"
$Tmp = Join-Path $env:TEMP "fipmesh-setup"

function Step($msg) { Write-Host "`n==> $msg" -ForegroundColor Cyan }
function Fail($msg) { Write-Host "`nERROR: $msg" -ForegroundColor Red; exit 1 }

# Native commands run with $ErrorActionPreference = "Continue" (local to
# these functions): under "Stop", Windows PowerShell 5.1 turns anything a
# program writes to stderr (a pip warning, a Python traceback) into a
# terminating NativeCommandError. Success is judged by $LASTEXITCODE instead.

function Invoke-Checked {
    # Run a native command, showing its output; stop the script if it fails.
    param([string]$Exe, [string[]]$Arguments)
    $ErrorActionPreference = "Continue"
    & $Exe @Arguments
    if ($LASTEXITCODE -ne 0) { Fail "'$Exe $($Arguments -join ' ')' failed (exit $LASTEXITCODE)" }
}

function Invoke-Quiet {
    # Run a native command for its stdout, hiding stderr. Never stops the
    # script: check $LASTEXITCODE afterwards.
    param([string]$Exe, [string[]]$Arguments)
    $ErrorActionPreference = "Continue"
    & $Exe @Arguments 2>$null
}

function Expand-Download {
    # Download the first of $Urls that works, unzip it to a scratch folder and
    # return that folder. $Manual says how to do it by hand if none works.
    param([string[]]$Urls, [string]$Name, [string]$Manual)
    New-Item -ItemType Directory -Force -Path $Tmp | Out-Null
    $zip = Join-Path $Tmp "$Name.zip"
    $dir = Join-Path $Tmp $Name
    $done = $false
    foreach ($url in $Urls) {
        Write-Host "    downloading $url"
        try {
            # A non-browser user agent: SourceForge answers PowerShell's
            # default (it contains "Mozilla") with an HTML download page.
            Invoke-WebRequest -UseBasicParsing -UserAgent "Wget" -Uri $url -OutFile $zip
            $head = [IO.File]::ReadAllBytes($zip)[0..1]
            if ($head[0] -ne 0x50 -or $head[1] -ne 0x4B) { throw "got a web page, not a zip file" }
            $done = $true
            break
        } catch {
            Write-Host "    failed: $($_.Exception.Message)" -ForegroundColor Yellow
        }
    }
    if (-not $done) { Fail "could not download $Name from any source. $Manual" }
    if (Test-Path $dir) { Remove-Item -Recurse -Force $dir }
    Expand-Archive -Path $zip -DestinationPath $dir
    Remove-Item -Force $zip
    return $dir
}

# -- 1. Python 3.12 --
Step "Finding Python 3.12"
$BasePy = $null
$candidates = @(
    @{ Exe = "py";      Args = @("-3.12") },
    @{ Exe = "python";  Args = @() },
    @{ Exe = "python3"; Args = @() }
)
foreach ($cand in $candidates) {
    if (-not (Get-Command $cand.Exe -ErrorAction SilentlyContinue)) { continue }
    $ver = Invoke-Quiet $cand.Exe ($cand.Args + @("-c", "import sys; print('%d.%d' % sys.version_info[:2])"))
    if ($LASTEXITCODE -eq 0 -and $ver -match '^3\.(9|10|11|12)$') {
        $BasePy = $cand
        Write-Host "    using $($cand.Exe) $($cand.Args -join ' ') (Python $ver)"
        break
    }
}
if (-not $BasePy) {
    Fail ("Python 3.9-3.12 not found (open3d has no wheels for 3.13+). Install Python 3.12 for " +
          "your user only, no admin needed:`n" +
          "    winget install -e --id Python.Python.3.12 --scope user`n" +
          "or the python.org installer with 'Install for all users' UNticked. Then re-run this script.")
}

# -- 2. venv + Python packages --
Step "Creating the virtualenv (venv\)"
if (-not (Test-Path $VenvPy)) {
    Invoke-Checked $BasePy.Exe ($BasePy.Args + @("-m", "venv", $Venv))
} else {
    Write-Host "    already exists"
}
Invoke-Checked $VenvPy @("-m", "pip", "install", "--upgrade", "pip", "wheel")

Step "Installing the CUDA build of torch"
Invoke-Quiet $VenvPy @("-c", "import torch, sys; sys.exit(0 if torch.version.cuda else 1)") | Out-Null
if ($LASTEXITCODE -eq 0) {
    Write-Host "    already installed"
} else {
    $ok = $false
    foreach ($cu in $TorchCuda) {
        Write-Host "    trying https://download.pytorch.org/whl/$cu"
        $ErrorActionPreference = "Continue"
        & $VenvPy -m pip install --upgrade --force-reinstall torch --index-url "https://download.pytorch.org/whl/$cu"
        $ErrorActionPreference = "Stop"
        if ($LASTEXITCODE -eq 0) { $ok = $true; break }
    }
    if (-not $ok) { Fail "could not install a CUDA 12 build of torch (tried: $($TorchCuda -join ', '))" }
}

Step "Installing requirements.txt"
Invoke-Checked $VenvPy @("-m", "pip", "install", "-r", (Join-Path $Here "requirements.txt"))
# rembg or a dependency can drag in the CPU-only onnxruntime, which shares
# onnxruntime-gpu's package folder and silently replaces the GPU build.
$cpuOrt = Invoke-Quiet $VenvPy @("-m", "pip", "show", "onnxruntime")
if ($LASTEXITCODE -eq 0 -and $cpuOrt) {
    Write-Host "    removing CPU-only onnxruntime (it overrides onnxruntime-gpu)"
    Invoke-Checked $VenvPy @("-m", "pip", "uninstall", "-y", "onnxruntime")
    Invoke-Checked $VenvPy @("-m", "pip", "install", "--force-reinstall", "--no-deps", "onnxruntime-gpu<1.27")
}

# -- 3. COLMAP (CUDA build) --
$ColmapDir = Join-Path $Here "colmap"
if ($SkipColmap) {
    Step "Skipping COLMAP (-SkipColmap)"
} elseif ((Test-Path (Join-Path $ColmapDir "COLMAP.bat")) -and -not $Force) {
    Step "COLMAP already in colmap\"
} else {
    Step "Downloading COLMAP (Windows CUDA release)"
    $rel = Invoke-RestMethod -UseBasicParsing -Uri "https://api.github.com/repos/colmap/colmap/releases/latest" `
        -Headers @{ "User-Agent" = "fipmesh-setup" }
    $asset = $rel.assets | Where-Object { $_.name -match 'windows.*cuda\.zip$' -and $_.name -notmatch 'nocuda' } |
        Select-Object -First 1
    if (-not $asset) { Fail "no *windows-cuda.zip asset in COLMAP release $($rel.tag_name); download it by hand (see WINDOWS_SETUP.md)" }
    Write-Host "    release $($rel.tag_name): $($asset.name)"
    $dir = Expand-Download @($asset.browser_download_url) "colmap" `
        "Download colmap-x64-windows-cuda.zip from https://github.com/colmap/colmap/releases and unzip it so colmap\COLMAP.bat exists."
    $bat = Get-ChildItem -Path $dir -Recurse -Filter "COLMAP.bat" | Select-Object -First 1
    if (-not $bat) { Fail "COLMAP.bat not found in $($asset.name)" }
    if (Test-Path $ColmapDir) { Remove-Item -Recurse -Force $ColmapDir }
    # Copy, not Move-Item: %TEMP% and the repo can be on different drives.
    Copy-Item -Recurse -Path $bat.DirectoryName -Destination $ColmapDir
}

# -- 4. exiftool --
$ToolsDir = Join-Path $Here "tools"
$Exiftool = Join-Path $ToolsDir "exiftool.exe"
if ($SkipExiftool) {
    Step "Skipping exiftool (-SkipExiftool)"
} elseif ((Test-Path $Exiftool) -and -not $Force) {
    Step "exiftool already in tools\"
} else {
    Step "Downloading exiftool"
    $manual = ("Download the 64-bit Windows zip from https://exiftool.org, put 'exiftool(-k).exe' " +
               "(renamed to exiftool.exe) and its exiftool_files folder in tools\, then re-run.")
    try {
        $ver = ([string](Invoke-WebRequest -UseBasicParsing -Uri "https://exiftool.org/ver.txt").Content).Trim()
    } catch {
        Fail "could not read the current exiftool version ($($_.Exception.Message)). $manual"
    }
    # The zips live on SourceForge; exiftool.org itself no longer serves them
    # at a stable URL (404 as of 13.59), so it's only the fallback.
    $dir = Expand-Download @(
        "https://sourceforge.net/projects/exiftool/files/exiftool-${ver}_64.zip/download",
        "https://exiftool.org/exiftool-${ver}_64.zip"
    ) "exiftool" $manual
    $exe = Get-ChildItem -Path $dir -Recurse -Filter "exiftool*.exe" | Select-Object -First 1
    if (-not $exe) { Fail "exiftool .exe not found in exiftool-${ver}_64.zip" }
    New-Item -ItemType Directory -Force -Path $ToolsDir | Out-Null
    # Ship "exiftool(-k).exe" as exiftool.exe: the (-k) name makes it wait
    # for a keypress after every run, which would hang Stage 2.
    Copy-Item -Force $exe.FullName $Exiftool
    $files = Join-Path $exe.DirectoryName "exiftool_files"
    if (Test-Path $files) {
        $dest = Join-Path $ToolsDir "exiftool_files"
        if (Test-Path $dest) { Remove-Item -Recurse -Force $dest }
        Copy-Item -Recurse $files $dest
    }
    Write-Host "    exiftool $ver -> tools\exiftool.exe"
}

if (Test-Path $Tmp) { Remove-Item -Recurse -Force $Tmp -ErrorAction SilentlyContinue }

# -- 5. check --
Step "Checking the setup"
$ErrorActionPreference = "Continue"
& $VenvPy (Join-Path $Here "check_setup.py")
exit $LASTEXITCODE
