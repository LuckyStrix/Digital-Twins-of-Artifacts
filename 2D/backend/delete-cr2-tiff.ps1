# delete-cr2-tiff.ps1

# --- CONFIG ---
$rootPath = "C:\Users\erago\OneDrive\Documents\RIT\EFIP\papyrusPipelineFull\backend"   # <-- change this

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

$filesToDelete = Get-ChildItem -Path $rootPath -Recurse -Include *.cr2, *.tiff, *.glb -File |
    Where-Object { $excludeFiles -notcontains $_.Name }

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