param([Parameter(ValueFromRemainingArguments = $true)][string[]]$AppArguments)
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

try {
    $ProjectRoot = Split-Path -Parent $PSScriptRoot
    $ApplicationRoot = Join-Path $ProjectRoot 'beyond-possession'
    $Manifest = Get-Content -LiteralPath (Join-Path $PSScriptRoot 'python_windows.json') -Raw | ConvertFrom-Json
    $Architecture = if ($env:PROCESSOR_ARCHITEW6432) { $env:PROCESSOR_ARCHITEW6432 } else { $env:PROCESSOR_ARCHITECTURE }
    $Architecture = $Architecture.ToUpperInvariant()
    $Package = $Manifest.archives.$Architecture
    if (-not $Package) { throw "Unsupported Windows processor: $Architecture" }
    if ($Package.url -notmatch '^https://www\.python\.org/ftp/python/') { throw 'Unexpected Python download source.' }
    $RuntimeBase = Join-Path $ProjectRoot '.runtime'
    $Runtime = Join-Path $RuntimeBase ("python-" + $Manifest.version + '-' + $Architecture)
    $Python = Join-Path $Runtime 'python.exe'
    $Marker = Join-Path $Runtime 'verified.sha256'
    $Ready = (Test-Path -LiteralPath $Python) -and (Test-Path -LiteralPath $Marker)
    if ($Ready) { $Ready = (Get-Content -LiteralPath $Marker -Raw).Trim() -eq $Package.sha256 }
    if (-not $Ready) {
        Write-Host 'Preparing Python for this project. This download happens once.'
        New-Item -ItemType Directory -Path $RuntimeBase -Force | Out-Null
        $Temporary = Join-Path $RuntimeBase ([Guid]::NewGuid().ToString('N'))
        New-Item -ItemType Directory -Path $Temporary | Out-Null
        try {
            $Archive = Join-Path $Temporary 'python.zip'
            Invoke-WebRequest -Uri $Package.url -OutFile $Archive -UseBasicParsing
            $ActualHash = (Get-FileHash -LiteralPath $Archive -Algorithm SHA256).Hash.ToLowerInvariant()
            if ($ActualHash -ne $Package.sha256) { throw 'Python download verification failed. No runtime was installed.' }
            $Extracted = Join-Path $Temporary 'python'
            Expand-Archive -LiteralPath $Archive -DestinationPath $Extracted
            if (-not (Test-Path -LiteralPath (Join-Path $Extracted 'python.exe'))) { throw 'The verified archive does not contain Python.' }
            $PathFile = Get-ChildItem -LiteralPath $Extracted -Filter '*._pth' | Select-Object -First 1
            if (-not $PathFile) { throw 'Python runtime path configuration is missing.' }
            # The official embedded runtime isolates imports; add the project root.
            Add-Content -LiteralPath $PathFile.FullName -Value "`r`n$ProjectRoot`r`n$ApplicationRoot" -Encoding UTF8
            Set-Content -LiteralPath (Join-Path $Extracted 'verified.sha256') -Value $Package.sha256 -Encoding ASCII
            if (Test-Path -LiteralPath $Runtime) { Remove-Item -LiteralPath $Runtime -Recurse -Force }
            Move-Item -LiteralPath $Extracted -Destination $Runtime
        }
        finally {
            if (Test-Path -LiteralPath $Temporary) { Remove-Item -LiteralPath $Temporary -Recurse -Force }
        }
    }
    Set-Location -LiteralPath $ProjectRoot
    & $Python (Join-Path $ProjectRoot 'start.py') @AppArguments
    exit $LASTEXITCODE
}
catch {
    Write-Host ("Could not start Beyond Possession: " + $_.Exception.Message) -ForegroundColor Red
    Write-Host 'Extract the full ZIP into a writable folder and keep internet connected for the first launch.'
    exit 1
}
