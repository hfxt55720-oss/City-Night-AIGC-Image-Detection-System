$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Makensis = $env:MAKENSIS
if ([string]::IsNullOrWhiteSpace($Makensis)) {
    $Makensis = "makensis"
}
$SourceDir = Join-Path $ProjectRoot "dist\AIGC_Detector"
$InstallerRoot = Join-Path $ProjectRoot "installer"
$PayloadDir = Join-Path $InstallerRoot "payload\AIGC_Detector"
$ScriptPath = Join-Path $ProjectRoot "installer_nsis.nsi"
$SetupExe = Join-Path $InstallerRoot "AIGC_Detector_Setup.exe"
$SourceExe = Join-Path $SourceDir "AIGC_Detector.exe"

Set-Location $ProjectRoot

if ($Makensis -ne "makensis" -and -not (Test-Path $Makensis)) {
    throw "NSIS not found: $Makensis. Install NSIS in the aide environment first."
}
if (-not (Test-Path -Path $SourceExe)) {
    throw "PyInstaller dist directory not found: $SourceDir. Run build_release.ps1 first."
}

New-Item -ItemType Directory -Force -Path $InstallerRoot | Out-Null
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $PayloadDir) | Out-Null

robocopy $SourceDir $PayloadDir /E /COPY:DAT /R:2 /W:1 /NFL /NDL /NJH /NJS /NC /NS | Out-Host
$code = $LASTEXITCODE
if ($code -ge 8) {
    throw "Failed to copy installer payload. Robocopy exit code: $code"
}

& $Makensis /INPUTCHARSET UTF8 $ScriptPath
if (-not (Test-Path $SetupExe)) {
    throw "Installer was not generated: $SetupExe"
}

Write-Host ""
Write-Host "Installer generated: $SetupExe"
Write-Host "Distribute the whole directory: $InstallerRoot"
