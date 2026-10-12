$ErrorActionPreference='Stop'
$base='C:\Users\operator\Documents\QwenResearch\historical\outputs\qwen38-windows'
$run=Join-Path $base 'pi-serving-025'
if(Test-Path -LiteralPath $run){throw 'Log directory already exists'}
New-Item -ItemType Directory -Path $run | Out-Null
$command='C:\Windows\System32\cmd.exe /d /c C:\Windows\System32\wsl.exe -d Ubuntu -- python3 /home/operator/qwen38-5090/optimization-runs/pi-serving-025/exec-manifest.py /home/operator/qwen38-5090/optimization-runs/pi-serving-025/launch.json 1>"'+$run+'\stdout.log" 2>"'+$run+'\stderr.log"'
$startup=New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ShowWindow=[uint16]0}
$result=Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine=$command;CurrentDirectory=$run;ProcessStartupInformation=$startup}
$receipt=@{ProcessId=$result.ProcessId;ReturnValue=$result.ReturnValue;Directory=$run;Utc=[DateTime]::UtcNow.ToString('o')}
$receipt | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $run 'receipt.json')
$receipt | ConvertTo-Json -Compress
if($result.ReturnValue -ne 0){throw 'Controller creation failed'}
