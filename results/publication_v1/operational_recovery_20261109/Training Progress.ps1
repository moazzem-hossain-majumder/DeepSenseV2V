# Local companion for the current pilot. Read-only: never changes the training job.
param([switch]$TrayOnly)
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing
$root = $PSScriptRoot
function Read-SharedText($path) {
    # Allow the worker to atomically replace JSON while the monitor reads it.
    $stream = [System.IO.File]::Open($path, [System.IO.FileMode]::Open, [System.IO.FileAccess]::Read, ([System.IO.FileShare]::ReadWrite -bor [System.IO.FileShare]::Delete))
    $reader = New-Object System.IO.StreamReader($stream)
    try { return $reader.ReadToEnd() }
    finally { $reader.Dispose(); $stream.Dispose() }
}
$reportDir = Join-Path $root 'results\multiscenario_pilot_v1'
$activePath = Join-Path $root 'results\active_training.json'
$job = 'pilot'
$epochLimit = 3
$stageTitle = 'Pilot'
$modelNames = @('B1','P1','P3')
$unitName = 'epochs'
$queuePath = Join-Path $root 'results\unattended_queue.json'
if (Test-Path -LiteralPath $activePath) {
    $active = Read-SharedText $activePath | ConvertFrom-Json
    $reportDir = Join-Path $root $active.directory
    $job = $active.job
    $epochLimit = [int]$active.epochs
    $stageTitle = $active.title
    if ($active.models) { $modelNames = @($active.models) }
    if ($active.unit) { $unitName = $active.unit }
    if ($active.queue_file) { $queuePath = $active.queue_file }
}
$epochBudget = $modelNames.Count * $epochLimit
$mutex = New-Object System.Threading.Mutex($false, 'Local\DeepSensePilotProgressV1')
if (-not $mutex.WaitOne(0)) {
    Set-Content -LiteralPath (Join-Path $reportDir 'show_progress.flag') -Value 'show'
    exit
}
[System.Windows.Forms.Application]::EnableVisualStyles()
$form = New-Object System.Windows.Forms.Form
$form.Text = 'DeepSense Training Progress'
$form.Size = New-Object System.Drawing.Size(760, 620)
$form.MinimumSize = $form.Size
$form.StartPosition = 'CenterScreen'
$form.Font = New-Object System.Drawing.Font('Segoe UI', 11)
$form.BackColor = [System.Drawing.Color]::White
$title = New-Object System.Windows.Forms.Label
$title.SetBounds(24, 18, 690, 32)
$title.Text = 'DeepSense | Scenarios 36-39'
$title.Font = New-Object System.Drawing.Font('Segoe UI', 17, [System.Drawing.FontStyle]::Bold)
$form.Controls.Add($title)
$headline = New-Object System.Windows.Forms.Label
$headline.SetBounds(24, 62, 690, 54)
$form.Controls.Add($headline)
$bar = New-Object System.Windows.Forms.ProgressBar
$bar.SetBounds(24, 120, 690, 22)
$form.Controls.Add($bar)
$details = New-Object System.Windows.Forms.TextBox
$details.SetBounds(24, 158, 690, 330)
$details.Multiline = $true
$details.ReadOnly = $true
$details.ScrollBars = 'Vertical'
$details.BackColor = [System.Drawing.Color]::White
$form.Controls.Add($details)
$footer = New-Object System.Windows.Forms.Label
$footer.SetBounds(24, 502, 690, 65)
$footer.Text = "Refreshes every 15 seconds. Closing this window keeps notifications active in the tray.`r`nRight-click the tray icon to exit. Training continues independently."
$form.Controls.Add($footer)
$tray = New-Object System.Windows.Forms.NotifyIcon
$tray.Icon = [System.Drawing.SystemIcons]::Information
$tray.Text = 'DeepSense training progress'
$tray.Visible = $true
$menu = New-Object System.Windows.Forms.ContextMenuStrip
$open = $menu.Items.Add('Show progress')
$quit = $menu.Items.Add('Exit progress monitor')
$tray.ContextMenuStrip = $menu
$script:allowExit = $false
$show = { $form.Show(); $form.WindowState = 'Normal'; $form.Activate() }
$open.add_Click($show)
$tray.add_DoubleClick($show)
$quit.add_Click({ $script:allowExit = $true; $form.Close() })
$form.add_FormClosing({ param($sender, $eventArgs)
    if (-not $script:allowExit) { $eventArgs.Cancel = $true; $form.Hide() }
})
$script:lastAlert = ''
$receipt = Join-Path $reportDir 'notification_receipt.txt'
if (Test-Path -LiteralPath $receipt) { $script:lastAlert = (Read-SharedText $receipt).Trim() }
function Read-Json($path) {
    if (Test-Path -LiteralPath $path) { return Read-SharedText $path | ConvertFrom-Json }
    return $null
}
function Update-Progress {
    try {
        if (Test-Path -LiteralPath $activePath) {
            $latest = Read-SharedText $activePath | ConvertFrom-Json
            $newDirectory = Join-Path $root $latest.directory
            if ($newDirectory -ne $script:reportDir) {
                $script:reportDir = $newDirectory
                $script:job = $latest.job
                $script:epochLimit = [int]$latest.epochs
                $script:stageTitle = $latest.title
                $script:modelNames = @($latest.models)
                $script:unitName = $latest.unit
                if ($latest.queue_file) { $script:queuePath = $latest.queue_file }
                $script:epochBudget = $modelNames.Count * $epochLimit
                $script:receipt = Join-Path $reportDir 'notification_receipt.txt'
                $script:lastAlert = ''
                if (Test-Path -LiteralPath $receipt) { $script:lastAlert = (Read-SharedText $receipt).Trim() }
            }
        }
        $showFlag = Join-Path $reportDir 'show_progress.flag'
        if (Test-Path -LiteralPath $showFlag) {
            Remove-Item -LiteralPath $showFlag
            & $show
        }
        $state = Read-Json (Join-Path $reportDir 'status.json')
        $supervisor = Read-Json (Join-Path $root ("results\data_acquisition\${job}_supervisor.json"))
        if ($null -eq $state) { $headline.Text = 'Waiting for the training status file...'; return }
        $age = ((Get-Date).ToUniversalTime() - [datetime]::Parse($state.updated_utc).ToUniversalTime()).TotalSeconds
        $heartbeatAge = if ($supervisor.updated_utc) { ((Get-Date).ToUniversalTime() - [datetime]::Parse($supervisor.updated_utc).ToUniversalTime()).TotalSeconds } else { 9999 }
        $rows = New-Object System.Collections.Generic.List[string]
        $done = 0
        foreach ($model in $modelNames) {
            $historyPath = Join-Path $reportDir ($model + '_history.json')
            $history = @()
            if (Test-Path -LiteralPath $historyPath) {
                # Assign before wrapping: Windows PowerShell 5.1 emits JSON arrays
                # as one pipeline object, so @(Read-Json ...) nests the array.
                $parsedHistory = Read-SharedText $historyPath | ConvertFrom-Json
                $history = @($parsedHistory)
            }
            $done += $history.Count
            if ($history.Count -gt 0) {
                $last = $history[-1]
                if ($model -eq 'analysis') { $rows.Add('Analysis completed; see this stage report.json / REPORT.md for results.') }
                else { $rows.Add(('{0}: {1}/{2} {5} | validation Top-1 {3:P2}, Top-5 {4:P2}' -f $model, $history.Count, $epochLimit, $last.val.top1, $last.val.top5, $unitName)) }
                if ($null -ne $last.val.macro_success5) { $rows.Add(('    PRIMARY: macro scenario success within 3 dB at 5 beams: {0:P2}' -f $last.val.macro_success5)) }
                $rows.Add(('    Last completed pass: {0:N1} seconds' -f $last.seconds))
                $finished = Read-Json (Join-Path $reportDir ($model + '_finished.json'))
                if ($finished) { $rows.Add('    Finished: ' + $finished.reason) }
            } else { $rows.Add($model + ': queued / no completed epoch yet') }
        }
        $phase = [string]$state.status
        if ($supervisor.status -eq 'needs_attention') { $phase = 'STOPPED - error needs attention' }
        elseif ($heartbeatAge -gt 90 -and $phase -notin @('complete','paused_by_user')) { $phase = 'MONITOR HEARTBEAT STALE - check worker' }
        elseif ($supervisor.status -eq 'retrying_in_15_seconds') { $phase = 'Retrying worker after interruption' }
        $headline.Text = "${stageTitle}: ${phase} | $($state.model) | Pass $($state.epoch)/$epochLimit`r`n$done of at most $epochBudget $unitName saved"
        if ($state.status -eq 'complete') {
            $headline.Text = "${stageTitle} complete | All queued tasks finished`r`n$done $unitName saved; early stopping may shorten training runs"
        }
        $bar.Value = if ($state.status -eq 'complete') { 100 } elseif ($state.batches) { [Math]::Min(100,[Math]::Max(0,[int](100*$state.batch/$state.batches))) } else { 0 }
        $lines = @("Current phase batch progress: $($state.batch) / $($state.batches) ($($bar.Value)%)", '')
        if ($state.status -eq 'complete') {
            $lines = @('Finished: ' + ([datetime]::Parse($state.updated_utc).ToLocalTime().ToString('yyyy-MM-dd HH:mm:ss')), 'No training job is running. Saved results are shown below.', '')
        }
        if ($state.status -in @('training','validating') -and $state.batch -gt 25) {
            $minutes = $state.elapsed_epoch_seconds / $state.batch * ($state.batches-$state.batch) / 60
            $lines += ('Approx. current pass remaining: {0:N1} minutes' -f $minutes)
        }
        $lines += @('', 'GPS / RGB baselines and fusion models; power_aware uses power-regret supervision.', '')
        $lines += $rows.ToArray()
        if ($job -in @('queue','publication')) {
            $queue = Read-Json $queuePath
            $lines += @('', 'AUTOMATIC QUEUE: ' + $queue.status)
            foreach ($entry in $queue.stages.PSObject.Properties) { $lines += ($entry.Value.title + ': ' + $entry.Value.status) }
            if ($queue.status -eq 'needs_attention') {
                $failedStage = $queue.stages.PSObject.Properties | Where-Object { $_.Name -eq [string]$queue.active_stage }
                $lines = @('QUEUE STOPPED: ' + $failedStage.Value.error, '') + $lines
            }
        }
        $lines += @('', 'Development results only. Calibration and test are reserved.',
            "Status updated: $([datetime]::Parse($state.updated_utc).ToLocalTime()) | age $([int]$age)s",
            "Supervisor: $($supervisor.status) | heartbeat age $([int]$heartbeatAge)s",
            'Queue timing is not proportional to epoch count; RGB models take longer.')
        if ($state.error) { $lines += "Error: $($state.error)" }
        $details.Text = $lines -join "`r`n"
        $alert = if ($state.status -eq 'complete') { 'complete' } elseif ($state.status -eq 'failed') { 'failed:' + $state.error } elseif ($supervisor.status -eq 'needs_attention') { 'needs_attention' } elseif ($heartbeatAge -gt 90 -and $state.status -ne 'paused_by_user') { 'stale' } else { '' }
        if ($alert -and $alert -ne $script:lastAlert) {
            $tray.BalloonTipTitle = if ($alert -eq 'complete') { "DeepSense $stageTitle completed" } else { 'DeepSense training needs attention' }
            $tray.BalloonTipText = if ($alert -eq 'complete') { 'All tasks in the selected stage finished. Open Training Progress for results. Final test evaluation is still pending.' } else { 'Open Training Progress for status. The supervisor may be retrying the worker.' }
            if ($alert -eq 'complete' -and $job -eq 'publication') { $tray.BalloonTipText = 'This research stage completed. The queue advances automatically where gates permit. Check the final evidence report for scientific readiness and held-out evaluation status.' }
            $tray.ShowBalloonTip(15000)
            [System.Media.SystemSounds]::Asterisk.Play()
            $script:lastAlert = $alert
            Set-Content -LiteralPath $receipt -Value $alert
            if ($alert -eq 'complete') { & $show }
        }
        Set-Content -LiteralPath (Join-Path $reportDir 'progress_monitor_heartbeat.txt') -Value (Get-Date).ToString('o')
    } catch { $details.Text = 'Unable to refresh; retrying in 15 seconds: ' + $_.Exception.Message }
}
$timer = New-Object System.Windows.Forms.Timer
$timer.Interval = 15000
$timer.add_Tick({ Update-Progress })
$form.add_Shown({ Update-Progress; if ($TrayOnly) { $form.Hide() } })
$timer.Start()
try { [System.Windows.Forms.Application]::Run($form) }
finally { $timer.Dispose(); $tray.Dispose(); $mutex.ReleaseMutex(); $mutex.Dispose() }
