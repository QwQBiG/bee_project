param(
    [string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot),
    [switch]$ForcePythonRefresh,
    [switch]$VerifyOnly
)

$ErrorActionPreference = "Stop"

$PythonVersion = "3.13.15"
$PythonArchiveName = "python-$PythonVersion-embed-amd64.zip"
$PythonArchiveUrl = "https://www.python.org/ftp/python/$PythonVersion/$PythonArchiveName"
$PythonArchiveSha256 = "d1f04d990aee1253d8569e8e5104e30fa9f5fa830899f14843448872d936a2cf"
$GetPipUrl = "https://bootstrap.pypa.io/get-pip.py"
$TorchWheelName = "torch-2.13.0+cu132-cp313-cp313-win_amd64.whl"
$TorchVersion = "2.13.0+cu132"
$TorchvisionVersion = "0.28.0+cu132"
$TorchvisionIndex = "https://download.pytorch.org/whl/cu132"

$ProjectRoot = [System.IO.Path]::GetFullPath($ProjectRoot)
$RuntimeRoot = Join-Path $ProjectRoot ".runtime"
$PythonRoot = Join-Path $RuntimeRoot "python313"
$PythonExe = Join-Path $PythonRoot "python.exe"
$PackagesRoot = Join-Path $ProjectRoot "packages"
$RequirementsPath = Join-Path $ProjectRoot "requirements.txt"
$PythonArchivePath = Join-Path $PackagesRoot $PythonArchiveName
$GetPipPath = Join-Path $PackagesRoot "get-pip.py"
$TorchWheelPath = Join-Path $PackagesRoot $TorchWheelName

function Invoke-Checked {
    param(
        [string]$Executable,
        [string[]]$Arguments,
        [string]$FailureMessage
    )
    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$FailureMessage (exit code $LASTEXITCODE)"
    }
}

function Test-PortablePython {
    if (-not (Test-Path -LiteralPath $PythonExe -PathType Leaf)) {
        return $false
    }
    & $PythonExe -c "import struct, sys; raise SystemExit(0 if sys.version_info[:3] == (3, 13, 15) and struct.calcsize('P') == 8 else 1)" *> $null
    return $LASTEXITCODE -eq 0
}

function Get-VerifiedPythonArchive {
    New-Item -ItemType Directory -Force -Path $PackagesRoot | Out-Null

    $archiveIsValid = $false
    if (Test-Path -LiteralPath $PythonArchivePath -PathType Leaf) {
        $archiveHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $PythonArchivePath).Hash.ToLowerInvariant()
        $archiveIsValid = $archiveHash -eq $PythonArchiveSha256
        if (-not $archiveIsValid) {
            Write-Host "[Bee Vision] Cached Python archive has an invalid SHA-256 hash; downloading a clean copy."
        }
    }

    if (-not $archiveIsValid) {
        $downloadPath = "$PythonArchivePath.download-$PID"
        try {
            Write-Host "[Bee Vision] Downloading official Python $PythonVersion embeddable package..."
            Invoke-WebRequest -UseBasicParsing -Uri $PythonArchiveUrl -OutFile $downloadPath
            $downloadHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $downloadPath).Hash.ToLowerInvariant()
            if ($downloadHash -ne $PythonArchiveSha256) {
                throw "Downloaded Python archive failed SHA-256 verification."
            }
            Move-Item -LiteralPath $downloadPath -Destination $PythonArchivePath -Force
        } finally {
            if (Test-Path -LiteralPath $downloadPath) {
                Remove-Item -LiteralPath $downloadPath -Force
            }
        }
    }
}

function Initialize-PortablePython {
    Get-VerifiedPythonArchive
    New-Item -ItemType Directory -Force -Path $RuntimeRoot | Out-Null

    $stagingRoot = Join-Path $RuntimeRoot "python313-staging-$PID"
    if (Test-Path -LiteralPath $stagingRoot) {
        Remove-Item -LiteralPath $stagingRoot -Recurse -Force
    }
    New-Item -ItemType Directory -Path $stagingRoot | Out-Null

    try {
        Write-Host "[Bee Vision] Extracting portable Python into .runtime/python313..."
        Expand-Archive -LiteralPath $PythonArchivePath -DestinationPath $stagingRoot -Force

        $pthFile = Get-ChildItem -LiteralPath $stagingRoot -Filter "python313._pth" | Select-Object -First 1
        if (-not $pthFile) {
            throw "The official Python archive does not contain python313._pth."
        }
        $pthLines = @(Get-Content -LiteralPath $pthFile.FullName)
        $pthLines = @($pthLines | ForEach-Object {
            if ($_ -match "^\s*#\s*import site\s*$") { "import site" } else { $_ }
        })
        if ($pthLines -notcontains "Lib\site-packages") {
            $pthLines += "Lib\site-packages"
        }
        Set-Content -LiteralPath $pthFile.FullName -Value $pthLines -Encoding ASCII
        New-Item -ItemType Directory -Force -Path (Join-Path $stagingRoot "Lib\site-packages") | Out-Null

        if (-not (Test-Path -LiteralPath $GetPipPath -PathType Leaf)) {
            Write-Host "[Bee Vision] Downloading the official pip bootstrap script..."
            Invoke-WebRequest -UseBasicParsing -Uri $GetPipUrl -OutFile $GetPipPath
        }
        $stagingPython = Join-Path $stagingRoot "python.exe"
        Invoke-Checked $stagingPython @($GetPipPath, "--no-warn-script-location") `
            "pip initialization failed"

        if (Test-Path -LiteralPath $PythonRoot) {
            $backupRoot = Join-Path $RuntimeRoot ("python313-backup-" + (Get-Date -Format "yyyyMMdd-HHmmss"))
            Move-Item -LiteralPath $PythonRoot -Destination $backupRoot
            Write-Host "[Bee Vision] Previous runtime was preserved at $backupRoot"
        }
        Move-Item -LiteralPath $stagingRoot -Destination $PythonRoot
    } catch {
        if (Test-Path -LiteralPath $stagingRoot) {
            Remove-Item -LiteralPath $stagingRoot -Recurse -Force
        }
        throw
    }

    if (-not (Test-PortablePython)) {
        throw "Portable Python initialization did not produce a usable Python $PythonVersion runtime."
    }
}

function Get-InstalledDistributionVersion {
    param([string]$DistributionName)
    $version = & $PythonExe -c "import importlib.metadata; print(importlib.metadata.version('$DistributionName'))" 2>$null
    if ($LASTEXITCODE -ne 0) {
        return $null
    }
    return ($version | Select-Object -Last 1).Trim()
}

function Install-ProjectDependencies {
    if (-not (Test-Path -LiteralPath $TorchWheelPath -PathType Leaf)) {
        throw "Required PyTorch wheel is missing: $TorchWheelPath"
    }
    if (-not (Test-Path -LiteralPath $RequirementsPath -PathType Leaf)) {
        throw "Requirements file is missing: $RequirementsPath"
    }

    $installedTorch = Get-InstalledDistributionVersion "torch"
    if ($installedTorch -ne $TorchVersion) {
        Write-Host "[Bee Vision] Installing $TorchWheelName..."
        Invoke-Checked $PythonExe @(
            "-m", "pip", "install", "--disable-pip-version-check",
            "--no-warn-script-location", $TorchWheelPath
        ) "Local PyTorch installation failed"
    } else {
        Write-Host "[Bee Vision] Reusing local PyTorch $installedTorch."
    }

    $installedTorchvision = Get-InstalledDistributionVersion "torchvision"
    if ($installedTorchvision -ne $TorchvisionVersion) {
        Write-Host "[Bee Vision] Installing torchvision $TorchvisionVersion from the CUDA 13.2 index..."
        Invoke-Checked $PythonExe @(
            "-m", "pip", "install", "--disable-pip-version-check",
            "--no-warn-script-location", "--no-deps", "--index-url", $TorchvisionIndex,
            "torchvision==$TorchvisionVersion"
        ) "CUDA torchvision installation failed"
    } else {
        Write-Host "[Bee Vision] Reusing torchvision $installedTorchvision."
    }

    Write-Host "[Bee Vision] Installing all remaining project dependencies..."
    Invoke-Checked $PythonExe @(
        "-m", "pip", "install", "--disable-pip-version-check",
        "--no-warn-script-location", "-r", $RequirementsPath
    ) "Project dependency installation failed"
}

function Test-ProjectRuntime {
    $verificationCode = @'
import json
import cv2
import matplotlib
import numpy
import pandas
import scipy
import torch
import torchvision
import ultralytics
import yaml

cuda_available = bool(torch.cuda.is_available())
device_name = torch.cuda.get_device_name(0) if cuda_available else None
if torch.__version__ != '2.13.0+cu132':
    raise RuntimeError(f'Expected torch 2.13.0+cu132, got {torch.__version__}')
if torchvision.__version__ != '0.28.0+cu132':
    raise RuntimeError(f'Expected torchvision 0.28.0+cu132, got {torchvision.__version__}')
if not cuda_available:
    raise RuntimeError('CUDA is unavailable; install a CUDA 13.2 compatible NVIDIA driver')
torch.zeros(1, device='cuda:0')
print(json.dumps({
    'python': __import__('sys').version.split()[0],
    'torch': torch.__version__,
    'torchvision': torchvision.__version__,
    'cuda_runtime': torch.version.cuda,
    'cuda_available': cuda_available,
    'device_name': device_name,
}, ensure_ascii=True))
'@
    Write-Host "[Bee Vision] Verifying imports and PyTorch runtime..."
    Invoke-Checked $PythonExe @("-c", $verificationCode) "Runtime verification failed"
}

if ($env:OS -ne "Windows_NT" -or -not [Environment]::Is64BitOperatingSystem) {
    throw "This portable runtime supports only 64-bit Windows."
}

if ($ForcePythonRefresh -or -not (Test-PortablePython)) {
    if ($VerifyOnly) {
        throw "Portable Python is not ready; run setup_runtime.bat first."
    }
    Initialize-PortablePython
} else {
    Write-Host "[Bee Vision] Reusing .runtime/python313/python.exe."
}

if (-not $VerifyOnly) {
    Install-ProjectDependencies
}
Test-ProjectRuntime

$runtimeInfo = [ordered]@{
    python = $PythonVersion
    python_path = ".runtime/python313/python.exe"
    torch = $TorchVersion
    torchvision = $TorchvisionVersion
    torch_wheel = "packages/$TorchWheelName"
    prepared_at = (Get-Date).ToUniversalTime().ToString("o")
}
$runtimeInfo | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $RuntimeRoot "runtime.json") -Encoding UTF8

Write-Host ""
Write-Host "[Bee Vision] Portable runtime is ready."
Write-Host "[Bee Vision] Command: run_cli.bat --help"
Write-Host "[Bee Vision] The whole project directory can now be copied to another 64-bit Windows PC."
