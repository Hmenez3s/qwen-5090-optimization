#requires -Version 5.1
[CmdletBinding()]
param()

Set-StrictMode -Version 2.0
$ErrorActionPreference = 'Stop'

$sourceDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$sourceExe = Join-Path $sourceDir 'qwenctl.exe'
if (-not (Test-Path -LiteralPath $sourceExe -PathType Leaf)) {
    $sourceExe = Join-Path $sourceDir 'dist\qwenctl.exe'
}
$sourceReadme = Join-Path $sourceDir 'README.md'
$localAppData = [Environment]::GetFolderPath('LocalApplicationData')
if ([string]::IsNullOrWhiteSpace($localAppData)) {
    throw 'LOCALAPPDATA não está disponível.'
}
$destination = Join-Path $localAppData 'Programs\QwenCtl'
$configDir = Join-Path $localAppData 'QwenCtl'
$configPath = Join-Path $configDir 'config.json'
$configExisted = Test-Path -LiteralPath $configPath -PathType Leaf

if (-not (Test-Path -LiteralPath $sourceExe -PathType Leaf)) {
    throw "Executável não encontrado: $sourceExe. Execute o build antes da instalação."
}
if (-not (Test-Path -LiteralPath $sourceReadme -PathType Leaf)) {
    throw "README não encontrado: $sourceReadme."
}

New-Item -ItemType Directory -Force -Path $destination | Out-Null
Copy-Item -LiteralPath $sourceExe -Destination (Join-Path $destination 'qwenctl.exe') -Force
Copy-Item -LiteralPath $sourceReadme -Destination (Join-Path $destination 'README.md') -Force

# The configuration is user-owned data.  This installer never creates,
# replaces, or removes config.json.
if (-not (Test-Path -LiteralPath $configDir -PathType Container)) {
    # Only the parent data directory is created when it does not exist; no
    # configuration file is generated here.
    New-Item -ItemType Directory -Force -Path $configDir | Out-Null
}
if ($configExisted -and -not (Test-Path -LiteralPath $configPath -PathType Leaf)) {
    throw 'config.json deixou de existir durante a instalação.'
}

$userPath = [Environment]::GetEnvironmentVariable('Path', 'User')
$entries = @()
if (-not [string]::IsNullOrWhiteSpace($userPath)) {
    $entries = @($userPath -split ';' | Where-Object { $_ -ne '' })
}
$alreadyPresent = $false
foreach ($entry in $entries) {
    if ([string]::Equals($entry.TrimEnd('\'), $destination.TrimEnd('\'), [StringComparison]::OrdinalIgnoreCase)) {
        $alreadyPresent = $true
        break
    }
}
if (-not $alreadyPresent) {
    $newPath = (($entries + $destination) -join ';')
    [Environment]::SetEnvironmentVariable('Path', $newPath, 'User')
}

Write-Host "qwenctl instalado em $destination"
if ($alreadyPresent) {
    Write-Host 'PATH do usuário já continha o diretório (nenhuma duplicata criada).'
} else {
    Write-Host 'Diretório adicionado uma vez ao PATH do usuário.'
}
Write-Host ''
Write-Host 'Próximos comandos:'
Write-Host '  qwenctl version'
Write-Host '  qwenctl status'
Write-Host '  qwenctl health'
Write-Host '  qwenctl start'
