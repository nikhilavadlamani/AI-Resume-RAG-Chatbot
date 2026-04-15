$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot

if (Test-Path "$ProjectRoot\rag_env\Scripts\python.exe") {
    $PythonExe = "$ProjectRoot\rag_env\Scripts\python.exe"
} elseif (Test-Path "$ProjectRoot\venv\Scripts\python.exe") {
    $PythonExe = "$ProjectRoot\venv\Scripts\python.exe"
} else {
    $PythonExe = "python"
}

Set-Location $ProjectRoot
& $PythonExe -m uvicorn app.main:app --reload
