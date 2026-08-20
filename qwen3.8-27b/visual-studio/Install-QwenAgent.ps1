[CmdletBinding(SupportsShouldProcess)]
param(
    [ValidateSet("Repository", "User")]
    [string]$Scope = "Repository",

    [string]$RepositoryPath = (Get-Location).Path
)

$ErrorActionPreference = "Stop"
$source = Join-Path $PSScriptRoot "qwen38-local.agent.md"

if ($Scope -eq "User") {
    $destinationDirectory = Join-Path $env:USERPROFILE ".github\agents"
}
else {
    $destinationDirectory = Join-Path $RepositoryPath ".github\agents"
}

$destination = Join-Path $destinationDirectory "qwen38-local.agent.md"
if ($PSCmdlet.ShouldProcess($destination, "Install Qwen custom agent")) {
    New-Item -ItemType Directory -Force -Path $destinationDirectory | Out-Null
    Copy-Item -Force -Path $source -Destination $destination
    Write-Host "Installed Qwen custom agent at $destination"
}
