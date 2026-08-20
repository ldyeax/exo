[CmdletBinding()]
param(
    [Parameter(Mandatory)]
    [string]$HostName,

    [string]$UserName = "root",

    [int]$LocalPort = 11434,

    [int]$RemotePort = 11434
)

$ErrorActionPreference = "Stop"
Write-Host "Forwarding http://127.0.0.1:$LocalPort to $UserName@$HostName"
Write-Host "Keep this window open while Visual Studio uses Qwen."

& ssh -N `
    -o ExitOnForwardFailure=yes `
    -L "${LocalPort}:127.0.0.1:${RemotePort}" `
    "${UserName}@${HostName}"

if ($LASTEXITCODE -ne 0) {
    throw "ssh tunnel exited with code $LASTEXITCODE"
}
