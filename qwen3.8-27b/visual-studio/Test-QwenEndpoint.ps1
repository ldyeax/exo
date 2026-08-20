[CmdletBinding()]
param(
    [string]$Endpoint = "http://127.0.0.1:11434",

    [string]$Model = "Qwen3.8-27B-FP8"
)

$ErrorActionPreference = "Stop"
$tags = Invoke-RestMethod -Method Get -Uri "$Endpoint/api/tags"
if ($Model -notin $tags.models.name) {
    throw "Model '$Model' was not returned by $Endpoint/api/tags"
}

$showBody = @{ model = $Model } | ConvertTo-Json
$show = Invoke-RestMethod `
    -Method Post `
    -Uri "$Endpoint/api/show" `
    -ContentType "application/json" `
    -Body $showBody
if ($show.model_info.'qwen3_5.context_length' -ne 262144) {
    throw "Expected context length 262144, received $($show.model_info.'qwen3_5.context_length')"
}

$tool = @{
    type = "function"
    function = @{
        name = "return_marker"
        description = "Return a marker through a tool call"
        parameters = @{
            type = "object"
            properties = @{
                marker = @{ type = "string" }
            }
            required = @("marker")
        }
    }
}
$chatBody = @{
    model = $Model
    stream = $false
    think = $false
    messages = @(
        @{
            role = "user"
            content = "Call return_marker once with marker VS2026_QWEN_OK. Do not answer in prose."
        }
    )
    tools = @($tool)
    options = @{
        temperature = 0
        num_predict = 128
    }
} | ConvertTo-Json -Depth 20

$chat = Invoke-RestMethod `
    -Method Post `
    -Uri "$Endpoint/api/chat" `
    -ContentType "application/json" `
    -Body $chatBody

$toolCalls = @($chat.message.tool_calls)
if ($toolCalls.Count -ne 1) {
    throw "Expected one tool call, received $($toolCalls.Count)"
}
if ($toolCalls[0].function.name -ne "return_marker") {
    throw "Unexpected tool call: $($toolCalls[0].function.name)"
}
if ($toolCalls[0].function.arguments.marker -ne "VS2026_QWEN_OK") {
    throw "Unexpected marker: $($toolCalls[0].function.arguments.marker)"
}

Write-Host "PASS: model discovery, 262144-token declaration, and tool calling"
Write-Host "Capabilities: $($show.capabilities -join ', ')"
