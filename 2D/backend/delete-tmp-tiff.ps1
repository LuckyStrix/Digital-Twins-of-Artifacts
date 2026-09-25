# delete-tmp-tiff.ps1

# --- CONFIG ---
$rootPath = $env:DELETE_TARGET_DIR

if ([string]::IsNullOrWhiteSpace($rootPath)) {
    Write-Host "ERROR: DELETE_TARGET_DIR environment variable is not set."
    exit 1
}

if (-not (Test-Path $rootPath)) {
    Write-Host "ERROR: Path does not exist: $rootPath"
    exit 1
}

# List the 9 files you want to KEEP (just the filename, or full path if you prefer exact matching)
$excludeFiles = @(
    "allLight.tiff",
    "eco.tiff",
    "ecross.tiff",
    "nco.tiff",
    "ncross.tiff",
    "sco.tiff",
    "scross.tiff",
    "wco.tiff",
    "wcross.tiff"
)

# --- SCRIPT ---
$dryRun = $false   # <-- set to $false once you've verified the file list below

# backend\calibration\ is protected wholesale. Its flat-field TIFFs happen to share
# the 9 capture filenames, but the colour-chart shot in calibration\chart\ does not
# -- deleting it would silently destroy the reference the colour matrix is fitted from.
$protectedDir = (Join-Path $rootPath "calibration") + [IO.Path]::DirectorySeparatorChar

$filesToDelete = Get-ChildItem -Path $rootPath -Recurse -Include *.tmp, *.tiff, *.glb -File |
    Where-Object { $excludeFiles -notcontains $_.Name } |
    Where-Object { -not $_.FullName.StartsWith($protectedDir, [StringComparison]::OrdinalIgnoreCase) }

Write-Host "Found $($filesToDelete.Count) files to delete (excluding $($excludeFiles.Count) protected files):"
$filesToDelete | ForEach-Object { Write-Host ("  " + $_.FullName) }

if (-not $dryRun) {
    $filesToDelete | Remove-Item -Force
    Write-Host ""
    Write-Host ("Deleted " + $filesToDelete.Count + " files.")
} else {
    Write-Host ""
    Write-Host "DRY RUN - no files were deleted. Review the list above, then set dryRun to false and rerun."
}