param(
    [ValidateSet("cpu", "cu118", "cu126", "cu128")]
    [string]$TorchBackend = "cu126",

    [ValidateSet("base", "deforming_plate", "dev", "none")]
    [string]$Profile = "dev",

    [string]$Venv = ".venv",

    [string]$PythonCommand = "py",

    [string[]]$PythonArgs = @("-3.11"),

    [switch]$SkipTorch
)

$ErrorActionPreference = "Stop"

function Invoke-Launcher {
    param([string[]]$ArgsList)
    $AllArgs = @($PythonArgs) + @($ArgsList)
    & $PythonCommand @AllArgs
}

function Invoke-VenvPython {
    param([string[]]$ArgsList)
    $VenvPython = Join-Path $Venv "Scripts/python.exe"
    & $VenvPython @ArgsList
}

if (-not (Test-Path $Venv)) {
    Write-Host "Creating virtual environment at $Venv"
    Invoke-Launcher @("-m", "venv", $Venv)
}

Write-Host "Upgrading packaging tools"
Invoke-VenvPython @("-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel")

if (-not $SkipTorch) {
    if ($TorchBackend -eq "cpu") {
        Write-Host "Installing PyTorch CPU wheels"
        Invoke-VenvPython @("-m", "pip", "install", "torch", "torchvision", "torchaudio")
    } else {
        $TorchIndex = "https://download.pytorch.org/whl/$TorchBackend"
        Write-Host "Installing PyTorch wheels from $TorchIndex"
        Invoke-VenvPython @(
            "-m", "pip", "install",
            "torch", "torchvision", "torchaudio",
            "--index-url", $TorchIndex
        )
    }
}

if ($Profile -ne "none") {
    $Requirements = "requirements/$Profile.txt"
    Write-Host "Installing dependencies from $Requirements"
    Invoke-VenvPython @("-m", "pip", "install", "-r", $Requirements)
}

Write-Host "Installing ValGraphNet in editable mode"
Invoke-VenvPython @("-m", "pip", "install", "-e", ".", "--no-deps")

Write-Host "Environment check"
Invoke-VenvPython @(
    "-c",
    "import torch; print('torch', torch.__version__, 'cuda_available=', torch.cuda.is_available(), 'cuda=', torch.version.cuda)"
)
Invoke-VenvPython @(
    "-c",
    "import torch_geometric, physicsnemo; print('pyg', torch_geometric.__version__); print('physicsnemo', physicsnemo.__version__)"
)

Write-Host ""
Write-Host "Done. Activate with:"
Write-Host "  .\$Venv\Scripts\Activate.ps1"
