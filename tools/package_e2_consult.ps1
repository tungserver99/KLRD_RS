# File-only packaging of existing artifacts. No Python, model, test or inference.
$ErrorActionPreference = 'Stop'
$repoRoot = [IO.Path]::GetFullPath((Split-Path -Parent $PSScriptRoot))
$stage = Join-Path $repoRoot 'verification/e2_team_consult_2026-11-02'
$archive = Join-Path $repoRoot 'KLRD_RS_E2_NUMERICAL_TEAM_REVIEW_2026-11-02.zip'
if (Test-Path -LiteralPath $archive) { throw 'Review archive already exists; refuse overwrite.' }
if (-not (Test-Path -LiteralPath (Join-Path $stage '00_DOC_TRUOC.md'))) { throw 'Missing Vietnamese explanation.' }
$latest = 'verification/vllm_server/2fac72a911234e0198799bc3fd878b38'
$selected = [ordered]@{}
foreach ($name in @('exact_sampler_parity','kv_cache_parity','batch_prefill_parity','reference_engine_parity','fp32_linear_shape_diagnostic')) {
    $selected[('01_latest_vllm/' + $name + '.json')] = ($latest + '/tests/' + $name + '.json')
}
$selected['01_latest_vllm/server_validation_error.json'] = ($latest + '/server_validation_error.json')
$selected['02_context/E1_continuation_summary.json'] = 'experiment1_rs_continuation/report/summary.json'
$selected['02_context/E1_frozen_lambda.json'] = 'experiment1_rs_continuation/teacher/global_lambda.json'
$selected['02_context/HF_run003_rotation.json'] = 'experiment2_rs/run_003/tests/exact_sampler_parity.json'
$selected['02_context/HF_run003_KV.json'] = 'experiment2_rs/run_003/tests/kv_cache_parity.json'
$selected['02_context/HF_run004_precision_probe.json'] = 'experiment2_rs/run_004/tests/kv_linear_precision_diagnostic.json'
$selected['02_context/HF_run005_rotation.json'] = 'experiment2_rs/run_005/tests/exact_sampler_parity.json'
$selected['02_context/HF_run005_KV.json'] = 'experiment2_rs/run_005/tests/kv_cache_parity.json'
foreach ($name in @('vllm_model','vllm_backend','vllm_gates')) {
    $selected[('03_source_snapshot/klrd_rs/e2/' + $name + '.py')] = ('klrd_rs/e2/' + $name + '.py')
}
$selected['03_source_snapshot/klrd_rs/math.py'] = 'klrd_rs/math.py'
$selected['03_source_snapshot/klrd_rs/readout.py'] = 'klrd_rs/readout.py'
$sourceIndex = @()
foreach ($entry in $selected.GetEnumerator()) {
    $source = Join-Path $repoRoot $entry.Value
    if (-not (Test-Path -LiteralPath $source -PathType Leaf)) { throw ('Missing source: ' + $entry.Value) }
    $target = Join-Path $stage $entry.Key
    [void][IO.Directory]::CreateDirectory((Split-Path -Parent $target))
    Copy-Item -LiteralPath $source -Destination $target
    $originalHash = (Get-FileHash -LiteralPath $source -Algorithm SHA256).Hash.ToLowerInvariant()
    if ((Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash.ToLowerInvariant() -ne $originalHash) { throw 'Copied bytes differ.' }
    $sourceIndex += [pscustomobject]@{ archive_path=$entry.Key; source_path=$entry.Value; sha256=$originalHash; bytes=(Get-Item -LiteralPath $source).Length }
}
$gateRows = foreach ($name in @('exact_sampler_parity','kv_cache_parity','batch_prefill_parity','reference_engine_parity')) {
    $report = Get-Content -LiteralPath (Join-Path $repoRoot ($latest + '/tests/' + $name + '.json')) -Raw -Encoding UTF8 | ConvertFrom-Json
    [pscustomobject]@{ gate=$name; status=$report.status; max_abs_logprob_error=$report.max_abs_logprob_error;
        p99_abs_logprob_error=$report.p99_abs_logprob_error; max_limit=0.0002; p99_limit=0.00005 }
}
$gateRows | Export-Csv -LiteralPath (Join-Path $stage 'gate_summary.csv') -NoTypeInformation -Encoding UTF8
$performance = @()
$timingSources = @()
$jobs = Get-ChildItem -LiteralPath (Join-Path $repoRoot 'experiment2_rs/run_005/private/jobs') -Filter 'primary_ownerclean_g00_chunk*.json' -File | Sort-Object Name | Select-Object -First 8
foreach ($job in $jobs) {
    $value = Get-Content -LiteralPath $job.FullName -Raw -Encoding UTF8 | ConvertFrom-Json
    $seconds = ($value.records | Measure-Object -Property amortized_generation_seconds -Sum).Sum
    $tokens = ($value.records | ForEach-Object { $_.generated_token_ids.Count } | Measure-Object -Sum).Sum
    $performance += [pscustomobject]@{ job=$job.BaseName; responses=@($value.records).Count; generation_seconds=$seconds;
        native_tokens=$tokens; max_new_tokens=512; scope='Partial HF generation batches; not independent ownership trials; excludes canonical verification time' }
    $timingSources += [pscustomobject]@{ source_path=('experiment2_rs/run_005/private/jobs/' + $job.Name);
        sha256=(Get-FileHash -LiteralPath $job.FullName -Algorithm SHA256).Hash.ToLowerInvariant(); content_included=$false; extracted_fields='count, duration, token count only' }
}
$performance | Export-Csv -LiteralPath (Join-Path $stage '02_context/HF_partial_generation_timing.csv') -NoTypeInformation -Encoding UTF8
$sourceDocument = [ordered]@{ package_date='2026-11-02 Australia/Sydney'; latest_diagnostic_run=$latest;
    result_scope='E2 numerical consultation only; no full study result'; source_snapshot_scope='Current source at packaging; exact executed-source hash unavailable for diagnostic runs';
    included_sources=$sourceIndex; private_timing_sources_not_included=$timingSources }
$sourceDocument | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $stage 'source_manifest.json') -Encoding UTF8
$files = @(Get-ChildItem -LiteralPath $stage -Recurse -File | Where-Object { $_.Name -ne 'SHA256_MANIFEST.json' })
$hashManifest = foreach ($file in $files) {
    $relative = $file.FullName.Substring($stage.Length + 1).Replace('\','/')
    [pscustomobject]@{ path=$relative; bytes=$file.Length; sha256=(Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash.ToLowerInvariant() }
}
$hashManifest | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $stage 'SHA256_MANIFEST.json') -Encoding UTF8
Add-Type -AssemblyName System.IO.Compression.FileSystem
[IO.Compression.ZipFile]::CreateFromDirectory($stage,$archive,[IO.Compression.CompressionLevel]::Optimal,$false)
$check = [IO.Compression.ZipFile]::OpenRead($archive)
try {
    if ($check.Entries.Count -ne ($files.Count + 1)) { throw 'ZIP entry count mismatch.' }
    foreach ($item in $hashManifest) {
        $entry = $check.GetEntry($item.path)
        if ($null -eq $entry) { throw ('Missing ZIP entry: ' + $item.path) }
        $stream = $entry.Open(); $algorithm = [Security.Cryptography.SHA256]::Create()
        try { $actual = [BitConverter]::ToString($algorithm.ComputeHash($stream)).Replace('-','').ToLowerInvariant() }
        finally { $stream.Dispose(); $algorithm.Dispose() }
        if ($actual -ne $item.sha256) { throw ('ZIP hash mismatch: ' + $item.path) }
    }
    foreach ($entry in $check.Entries) {
        if ($entry.FullName -match '(^|/)(private|keys|datasets|responses|weights)(/|$)') { throw 'Unexpected private artifact in ZIP.' }
    }
} finally { $check.Dispose() }
[pscustomobject]@{ zip=$archive; bytes=(Get-Item -LiteralPath $archive).Length; files=($files.Count+1);
    sha256=(Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash.ToLowerInvariant(); verification='All archived payload SHA256 checks match original staged files' } | ConvertTo-Json
