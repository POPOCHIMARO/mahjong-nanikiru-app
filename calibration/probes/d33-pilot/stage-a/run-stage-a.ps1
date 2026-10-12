# D.3.3 段階Aを実行する（設計 PHASE_D33_DESIGN.md 17.8節、18.6節）。
# 途中で止まっても、もう一度このスクリプトを実行すれば、済んだ鎖は残したまま残りだけを走らせる。
#
# 使い方（プロジェクトのどこからでも）:
#   powershell -NoProfile -File calibration\probes\d33-pilot\stage-a\run-stage-a.ps1
#   powershell -NoProfile -File calibration\probes\d33-pilot\stage-a\run-stage-a.ps1 -StartAt "2026-10-13T09:00:00"
param(
    # 開始する日時（このPCの時刻）。空ならすぐ始める。
    [string]$StartAt = "",
    # numpy などが入った Python。Codex 同梱のものを使う。
    [string]$Python = "$env:USERPROFILE\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
)
$ErrorActionPreference = "Stop"
$stageDir = $PSScriptRoot
$root = (Resolve-Path (Join-Path $stageDir "..\..\..\..")).Path
Set-Location $root
$outLog = Join-Path $stageDir "run.out.log"
$errLog = Join-Path $stageDir "run.err.log"

function Write-Log([string]$message) {
    "[$(Get-Date -Format s)] $message" | Add-Content -Path $outLog -Encoding utf8
}

# 実行中は、PCが無操作で自動的にスリープ・休止しないよう要求する。
# 電源の設定は変えない。このプロセスが終われば要求は消える（ふたを閉じる、電池で動かす場合は防げない）。
Add-Type -Namespace StageA -Name Power -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);'
[StageA.Power]::SetThreadExecutionState([uint32]"0x80000001") | Out-Null  # ES_CONTINUOUS + ES_SYSTEM_REQUIRED

if ($StartAt) {
    $wait = ([datetime]$StartAt - (Get-Date)).TotalSeconds
    if ($wait -gt 0) {
        Write-Log "waiting until $StartAt (pid $PID)"
        Start-Sleep -Seconds ([int][math]::Ceiling($wait))
    }
}

# 得点計算のライブラリ（mahjong 2.0.0）は、同梱のwheelから読む。
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONPATH = Join-Path $root "calibration\.dependency-cache\mahjong-2.0.0-py3-none-any.whl"

Write-Log "start (pid $PID)"
# 標準出力（鎖が1本済むごとの進み具合）と標準エラーを、ログの末尾に足していく。
cmd /c "`"$Python`" -B -m tools.calibrate_ev probe-policy-belief --stage stage-a >> `"$outLog`" 2>> `"$errLog`""
$code = $LASTEXITCODE
# 終了コード: 0＝完了、1＝上限の実時間を超えて停止、2＝manifestか入力の不一致
Write-Log "exit $code"
exit $code
