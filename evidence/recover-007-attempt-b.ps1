$ErrorActionPreference = 'Stop'
$base = 'C:\Users\operator\Documents\QwenResearch\historical\outputs\qwen38-windows'
$launcher = Join-Path $base 'start-shared-131k.ps1'
if (-not (Test-Path -LiteralPath $launcher)) { throw 'Preserved launcher missing' }
$run = Join-Path $base ('recovery-007-' + [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssZ'))
if (Test-Path -LiteralPath $run) { throw 'Recovery directory collision' }
New-Item -ItemType Directory -Path $run | Out-Null
New-Item -ItemType Directory -Path (Join-Path $run 'tmp') | Out-Null
$linux = '/mnt/c/' + $run.Substring(3).Replace('\','/')
$body = @"
`$ErrorActionPreference = 'Stop'
`$root = '$linux'
`$env:WSLENV = 'SGLANG_CACHE_DIR:FLASHINFER_WORKSPACE_BASE:TRITON_CACHE_DIR:TORCHINDUCTOR_CACHE_DIR:TORCH_EXTENSIONS_DIR:CUDA_CACHE_PATH:HF_HUB_OFFLINE:TRANSFORMERS_OFFLINE:HUGGINGFACE_HUB_CACHE:PYTHONDONTWRITEBYTECODE:TMPDIR'
`$env:SGLANG_CACHE_DIR = `$root + '/cache/sglang'
`$env:FLASHINFER_WORKSPACE_BASE = `$root
`$env:TRITON_CACHE_DIR = `$root + '/cache/triton'
`$env:TORCHINDUCTOR_CACHE_DIR = `$root + '/cache/torchinductor'
`$env:TORCH_EXTENSIONS_DIR = `$root + '/cache/torch-extensions'
`$env:CUDA_CACHE_PATH = `$root + '/cache/cuda'
`$env:HUGGINGFACE_HUB_CACHE = `$root + '/cache/huggingface'
`$env:HF_HUB_OFFLINE = '1'
`$env:TRANSFORMERS_OFFLINE = '1'
`$env:PYTHONDONTWRITEBYTECODE = '1'
`$env:TMPDIR = `$root + '/tmp'
& '$launcher'
exit `$LASTEXITCODE
"@
$encoded = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($body))
$command = 'C:\Windows\System32\cmd.exe /d /c C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe -NoProfile -NonInteractive -EncodedCommand ' + $encoded + ' 1>"' + $run + '\stdout.log" 2>"' + $run + '\stderr.log"'
$startup = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ShowWindow=[uint16]0}
$result = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine=$command; CurrentDirectory=$run; ProcessStartupInformation=$startup}
$receipt = @{Method='Win32_Process.Create'; ReturnValue=$result.ReturnValue; ProcessId=$result.ProcessId; Directory=$run; Launcher=$launcher; Utc=[DateTime]::UtcNow.ToString('o')}
$receipt | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $run 'receipt.json')
$receipt | ConvertTo-Json -Compress
if ($result.ReturnValue -ne 0) { throw ('Process creation failed: ' + $result.ReturnValue) }
