$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$PythonExe = $env:AIGC_PYTHON
if ([string]::IsNullOrWhiteSpace($PythonExe)) {
    $LocalVenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
    if (Test-Path $LocalVenvPython) {
        $PythonExe = $LocalVenvPython
    } else {
        $PythonExe = "python"
    }
}
$SpecFile = Join-Path $ProjectRoot "AIGC_Detector.spec"
$ReleaseDir = Join-Path $ProjectRoot "dist\AIGC_Detector"
$ExeFile = Join-Path $ReleaseDir "AIGC_Detector.exe"

Set-Location $ProjectRoot

if ($PythonExe -ne "python" -and -not (Test-Path $PythonExe)) {
    throw "Python环境不存在：$PythonExe"
}

& $PythonExe -m PyInstaller --version | Out-Host
& $PythonExe -m PyInstaller --clean --noconfirm $SpecFile

if (-not (Test-Path $ExeFile)) {
    throw "打包失败，未生成：$ExeFile"
}

Write-Host ""
Write-Host "打包完成：$ExeFile"
Write-Host "发布目录：$ReleaseDir"
