# Vesuvius Challenge workspace - activate with:  . C:\psn\env.ps1
# Everything (tools, Python, caches, venv, data) stays under C:\psn.
# Variables are set for the current shell session only; nothing is persisted.

$PSN = $PSScriptRoot

$env:UV_INSTALL_DIR        = "$PSN\.tools\uv"
$env:UV_CACHE_DIR          = "$PSN\.cache\uv"
$env:UV_PYTHON_INSTALL_DIR = "$PSN\.tools\python"
$env:UV_PYTHON_PREFERENCE  = "only-managed"   # never pick up the system Python 3.14
$env:UV_PROJECT_ENVIRONMENT = "$PSN\.venv"
$env:UV_NO_MODIFY_PATH     = "1"
$env:UV_TOOL_DIR           = "$PSN\.tools\uv-tools"
$env:UV_TOOL_BIN_DIR       = "$PSN\.tools\uv-tools\bin"

$env:PIP_CACHE_DIR         = "$PSN\.cache\pip"
$env:HF_HOME               = "$PSN\.cache\huggingface"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
$env:TORCH_HOME            = "$PSN\.cache\torch"
$env:XDG_CACHE_HOME        = "$PSN\.cache"
$env:MPLCONFIGDIR          = "$PSN\.cache\matplotlib"
$env:JUPYTER_CONFIG_DIR    = "$PSN\.cache\jupyter\config"
$env:JUPYTER_DATA_DIR      = "$PSN\.cache\jupyter\data"
$env:IPYTHONDIR            = "$PSN\.cache\ipython"
$env:NAPARI_CONFIG         = "$PSN\.cache\napari\settings.yaml"
$env:VESUVIUS_DATA         = "$PSN\data"

$env:PATH = "$PSN\.tools\uv;$PSN\.venv\Scripts;$env:PATH"

Write-Host "Vesuvius workspace active: $PSN" -ForegroundColor Green
