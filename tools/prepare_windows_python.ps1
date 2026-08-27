param(
    [string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot),
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

$PythonVersion = "3.13.15"
$PythonArchiveName = "python-$PythonVersion-embed-amd64.zip"
$PythonArchiveUrl = "https://www.python.org/ftp/python/$PythonVersion/$PythonArchiveName"
$PythonArchiveSha256 = "d1f04d990aee1253d8569e8e5104e30fa9f5fa830899f14843448872d936a2cf"
$GetPipUrl = "https://bootstrap.pypa.io/get-pip.py"

$ProjectRoot = [System.IO.Path]::GetFullPath($ProjectRoot)
$RuntimeDir = Join-Path $ProjectRoot ".runtime"
$PackagesDir = Join-Path $ProjectRoot "packages"
$RuntimePathFile = Join-Path $RuntimeDir "python-path.txt"
$PrimaryVenv = Join-Path $ProjectRoot ".venv"
$Python313Venv = Join-Path $ProjectRoot ".venv-py313"
$PortableDir = Join-Path $RuntimeDir "python313"

function Test-Python313 {
    param(
        [string]$PythonPath,
        [switch]$RequirePip
    )
    if (-not $PythonPath -or -not (Test-Path -LiteralPath $PythonPath -PathType Leaf)) {
        return $false
    }
    & $PythonPath -c "import struct, sys; raise SystemExit(0 if sys.version_info[:2] == (3, 13) and struct.calcsize('P') == 8 else 1)" *> $null
    if ($LASTEXITCODE -ne 0) {
        return $false
    }
    if ($RequirePip) {
        & $PythonPath -m pip --version *> $null
        return $LASTEXITCODE -eq 0
    }
    return $true
}

function Add-Candidate {
    param(
        [System.Collections.Generic.List[string]]$Candidates,
        [string]$Candidate
    )
    if (-not $Candidate) {
        return
    }
    $expanded = [Environment]::ExpandEnvironmentVariables($Candidate.Trim().Trim('"'))
    if (-not [System.IO.Path]::IsPathRooted($expanded)) {
        return
    }
    try {
        $fullPath = [System.IO.Path]::GetFullPath($expanded)
    } catch {
        return
    }
    if ($fullPath.StartsWith($ProjectRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
        return
    }
    if (-not $Candidates.Contains($fullPath)) {
        $Candidates.Add($fullPath)
    }
}

function Find-ExternalPython313 {
    $candidates = [System.Collections.Generic.List[string]]::new()

    foreach ($command in @(Get-Command python -All -ErrorAction SilentlyContinue)) {
        if ($command.CommandType -eq "Application") {
            Add-Candidate $candidates $command.Source
        }
    }

    if (Get-Command py -ErrorAction SilentlyContinue) {
        foreach ($line in @(& py -0p 2>$null)) {
            if ($line -match "([A-Za-z]:\\.+?python(?:3(?:\.13)?)?\.exe)\s*$") {
                Add-Candidate $candidates $Matches[1]
            }
        }
    }

    $knownPaths = @(
        "$env:LOCALAPPDATA\Programs\Python\Python313\python.exe",
        "$env:ProgramFiles\Python313\python.exe",
        "${env:ProgramFiles(x86)}\Python313\python.exe",
        "C:\Python313\python.exe"
    )
    foreach ($path in $knownPaths) {
        Add-Candidate $candidates $path
    }

    $registryRoots = @(
        "Registry::HKEY_CURRENT_USER\Software\Python\PythonCore\3.13\InstallPath",
        "Registry::HKEY_LOCAL_MACHINE\Software\Python\PythonCore\3.13\InstallPath",
        "Registry::HKEY_LOCAL_MACHINE\Software\WOW6432Node\Python\PythonCore\3.13\InstallPath"
    )
    foreach ($registryPath in $registryRoots) {
        if (-not (Test-Path -LiteralPath $registryPath)) {
            continue
        }
        $key = Get-Item -LiteralPath $registryPath
        Add-Candidate $candidates ($key.GetValue("ExecutablePath"))
        $installDir = $key.GetValue("")
        if ($installDir) {
            Add-Candidate $candidates (Join-Path $installDir "python.exe")
        }
    }

    foreach ($candidate in $candidates) {
        if (Test-Python313 $candidate) {
            return $candidate
        }
    }
    return $null
}

function Write-RuntimePath {
    param([string]$PythonPath)
    New-Item -ItemType Directory -Force -Path $RuntimeDir | Out-Null
    Set-Content -LiteralPath $RuntimePathFile -Value $PythonPath -Encoding ASCII
}

function New-ProjectVenv {
    param([string]$ExternalPython)

    foreach ($venvDir in @($PrimaryVenv, $Python313Venv)) {
        $venvPython = Join-Path $venvDir "Scripts\python.exe"
        if (Test-Python313 $venvPython -RequirePip) {
            return $venvPython
        }
    }

    $targetVenv = if (-not (Test-Path -LiteralPath $PrimaryVenv)) {
        $PrimaryVenv
    } else {
        $Python313Venv
    }
    Write-Host "[Bee Vision] Creating Python 3.13 environment: $targetVenv"
    & $ExternalPython -m venv $targetVenv
    if ($LASTEXITCODE -ne 0) {
        throw "Python 3.13 was found, but creating the project environment failed."
    }
    $venvPython = Join-Path $targetVenv "Scripts\python.exe"
    if (-not (Test-Python313 $venvPython -RequirePip)) {
        throw "The new project environment is not a usable 64-bit Python 3.13 environment."
    }
    return $venvPython
}

function Get-PortablePython313 {
    $portablePython = Join-Path $PortableDir "python.exe"
    if (Test-Python313 $portablePython -RequirePip) {
        return $portablePython
    }

    New-Item -ItemType Directory -Force -Path $RuntimeDir, $PackagesDir | Out-Null
    $archivePath = Join-Path $PackagesDir $PythonArchiveName
    $downloadNeeded = $true
    if (Test-Path -LiteralPath $archivePath -PathType Leaf) {
        $actualHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $archivePath).Hash.ToLowerInvariant()
        $downloadNeeded = $actualHash -ne $PythonArchiveSha256
        if ($downloadNeeded) {
            Write-Host "[Bee Vision] Cached portable Python archive failed SHA-256 verification; downloading it again."
        }
    }
    if ($downloadNeeded) {
        Write-Host "[Bee Vision] Downloading official portable Python $PythonVersion..."
        Invoke-WebRequest -UseBasicParsing -Uri $PythonArchiveUrl -OutFile $archivePath
    }
    $actualHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $archivePath).Hash.ToLowerInvariant()
    if ($actualHash -ne $PythonArchiveSha256) {
        throw "Portable Python archive SHA-256 verification failed."
    }

    $stagingDir = Join-Path $RuntimeDir "python313-staging-$PID"
    if (Test-Path -LiteralPath $stagingDir) {
        Remove-Item -LiteralPath $stagingDir -Recurse -Force
    }
    New-Item -ItemType Directory -Path $stagingDir | Out-Null
    Expand-Archive -LiteralPath $archivePath -DestinationPath $stagingDir -Force

    $pthFile = Get-ChildItem -LiteralPath $stagingDir -Filter "python313._pth" | Select-Object -First 1
    if (-not $pthFile) {
        throw "Portable Python archive does not contain python313._pth."
    }
    $pthLines = @(Get-Content -LiteralPath $pthFile.FullName)
    $pthLines = @($pthLines | ForEach-Object {
        if ($_ -match "^\s*#\s*import site\s*$") { "import site" } else { $_ }
    })
    if ($pthLines -notcontains "Lib\site-packages") {
        $pthLines += "Lib\site-packages"
    }
    Set-Content -LiteralPath $pthFile.FullName -Value $pthLines -Encoding ASCII
    New-Item -ItemType Directory -Force -Path (Join-Path $stagingDir "Lib\site-packages") | Out-Null

    $getPipPath = Join-Path $PackagesDir "get-pip.py"
    if (-not (Test-Path -LiteralPath $getPipPath -PathType Leaf)) {
        Write-Host "[Bee Vision] Downloading pip bootstrap..."
        Invoke-WebRequest -UseBasicParsing -Uri $GetPipUrl -OutFile $getPipPath
    }
    $stagingPython = Join-Path $stagingDir "python.exe"
    # PowerShell treats every success-stream item emitted by a function as part
    # of its return value. Send pip's progress directly to the host so that
    # Get-PortablePython313 returns only the Python executable path.
    & $stagingPython $getPipPath --no-warn-script-location | Out-Host
    if ($LASTEXITCODE -ne 0 -or -not (Test-Python313 $stagingPython -RequirePip)) {
        throw "Portable Python was extracted, but pip initialization failed."
    }

    if (Test-Path -LiteralPath $PortableDir) {
        $backupDir = Join-Path $RuntimeDir ("python313-incomplete-" + (Get-Date -Format "yyyyMMdd-HHmmss"))
        Move-Item -LiteralPath $PortableDir -Destination $backupDir
    }
    Move-Item -LiteralPath $stagingDir -Destination $PortableDir
    return $portablePython
}

if (-not [Environment]::Is64BitOperatingSystem) {
    throw "Bee Vision requires 64-bit Windows."
}

foreach ($venvDir in @($PrimaryVenv, $Python313Venv)) {
    $venvPython = Join-Path $venvDir "Scripts\python.exe"
    if (Test-Python313 $venvPython -RequirePip) {
        Write-Host "[Bee Vision] Reusing project Python 3.13: $venvPython"
        if (-not $DryRun) {
            Write-RuntimePath $venvPython
        }
        exit 0
    }
}

$externalPython = Find-ExternalPython313
if ($externalPython) {
    Write-Host "[Bee Vision] Found external Python 3.13: $externalPython"
    if ($DryRun) {
        exit 0
    }
    $runtimePython = New-ProjectVenv $externalPython
} else {
    Write-Host "[Bee Vision] External Python 3.13 was not found."
    if ($DryRun) {
        Write-Host "[Bee Vision] Portable Python $PythonVersion would be downloaded."
        exit 0
    }
    $runtimePython = Get-PortablePython313
}

Write-RuntimePath $runtimePython
Write-Host "[Bee Vision] Selected Python: $runtimePython"
