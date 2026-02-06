param(
    [Parameter(Mandatory = $true)]
    [string]$RepoRoot,

    [Parameter(Mandatory = $true)]
    [string]$DataRoot,

    [string]$RunRoot = (Join-Path "F:\databackup\GeoTGS-TC\output" ("Ablation_Stage2_" + (Get-Date -Format "yyyyMMdd_HHmmss"))),

    # 强烈建议显式传入 (Get-Command python).Source，避免误用 base python
    [string]$PythonExe = "",

    # Stage 选择（不要再从 CommonArgs 里塞 --from_step/--to_step，容易出你遇到的 bug）
    [string]$FromStep = "10",
    [string]$ToStep   = "12",
    [switch]$DryRun,

    # 共享 Stage1（RGB）+ 预处理（到 step9）并复用
    [switch]$ReuseStage1 = $true,
    [string]$SharedName = "_shared",

    # 训练/渲染分辨率与迭代
    [int]$RgbRes = 4,
    [int]$TRes   = 4,
    [int]$RgbIter = 30000,
    [int]$TIter   = 60000,

    # profiling：speed 只开 profile_pipeline；audit 才收 sizes/counts，并可选保存 logs
    [ValidateSet("speed","audit")]
    [string]$ProfileMode = "speed",
    [string[]]$ProfileLogsFor = @("debug_stats"),

    # 指标
    [switch]$EnablePlusMetrics,
    [int]$MetricsPlusK = 8,
    [ValidateSet(0,1)]
    [int]$MetricsPlusBg = 0,

    [switch]$EnableNovelMetrics,
    [int]$NovelViewN = 60,
    [ValidateSet(0,1)]
    [int]$NovelBg = 0,

    # 额外透传（仅建议放“不会破坏 token 边界”的参数；step/dry_run 用脚本参数）
    [string[]]$CommonArgs = @(),

    # 仅跑某些实验（可选）
    [string[]]$Only = @(),

    [switch]$SelfCheck
)

$ErrorActionPreference = "Stop"

function Quote-Arg([string]$a) {
    if ($null -eq $a) { return "" }
    if ($a -match '[\s"]') {
        return '"' + ($a -replace '"', '`"') + '"'
    }
    return $a
}

function Resolve-PythonExe() {
    if (-not [string]::IsNullOrWhiteSpace($PythonExe)) {
        return [IO.Path]::GetFullPath($PythonExe)
    }
    if ($env:CONDA_PREFIX) {
        $cand = Join-Path $env:CONDA_PREFIX "python.exe"
        if (Test-Path $cand) { return [IO.Path]::GetFullPath($cand) }
    }
    return (Get-Command python).Source
}

function Ensure-Junction([string]$LinkPath, [string]$TargetPath) {
    if (Test-Path $LinkPath) {
        # 删除 link 本身（不会删除 Target）
        Remove-Item $LinkPath -Recurse -Force
    }
    $lp = [IO.Path]::GetFullPath($LinkPath)
    $tp = [IO.Path]::GetFullPath($TargetPath)
    $null = cmd /c "mklink /J `"$lp`" `"$tp`""
    if (-not (Test-Path $lp)) {
        throw "Failed to create junction: $lp -> $tp"
    }
}

function Has-Any([string[]]$arr, [string]$val) {
    foreach ($x in $arr) { if ($x -eq $val) { return $true } }
    return $false
}

$RepoRoot = [IO.Path]::GetFullPath($RepoRoot)
$DataRoot = [IO.Path]::GetFullPath($DataRoot)
$RunRoot  = [IO.Path]::GetFullPath($RunRoot)

$Pipeline = Join-Path $RepoRoot "run_gtgs_full_pipeline.py"
$SummaryScript = Join-Path $RepoRoot "summarize_ablation_excel.py"

if (-not (Test-Path $Pipeline)) { throw "run_gtgs_full_pipeline.py not found: $Pipeline" }
if (-not (Test-Path $SummaryScript)) { Write-Warning "summarize_ablation_excel.py not found: $SummaryScript (will skip summary)" }

$py = Resolve-PythonExe

# 关键：防止你遇到的 torch 缺失（真实跑才强制）
if (-not $DryRun -and -not $SelfCheck) {
    try {
        & $py -c "import torch; print(torch.__version__)" | Out-Null
    } catch {
        throw "PythonExe cannot import torch. You are likely using a wrong python (e.g. base). Pass -PythonExe (Get-Command python).Source from your fgs env."
    }
}

New-Item -ItemType Directory -Force -Path $RunRoot | Out-Null

# profiling args（注意：collect_sizes 可能会拖慢，默认 speed 不开）
$profileArgs = @("--profile_pipeline", "--save_cmds")
if ($ProfileMode -eq "audit") {
    $profileArgs += @("--profile_collect_sizes", "--profile_collect_counts")
}

# metrics args
$metricArgs = @()
if ($EnablePlusMetrics) {
    $metricArgs += @("--run_metrics_plus", "--metrics_plus_K", "$MetricsPlusK", "--metrics_plus_bg", "$MetricsPlusBg")
}
if ($EnableNovelMetrics) {
    $metricArgs += @("--run_novel_view_metrics", "--novel_view_N", "$NovelViewN", "--novel_bg", "$NovelBg")
}

$dryArgs = @()
if ($DryRun -or $SelfCheck) { $dryArgs += @("--dry_run") }

# 统一基参
$baseArgs = @(
    "--data_root", $DataRoot,
    "--rgb_res", "$RgbRes",
    "--t_res", "$TRes",
    "--rgb_iter", "$RgbIter",
    "--t_iter", "$TIter"
) + $profileArgs + $metricArgs + $dryArgs

# ===== SelfCheck：尽量简单，跑一个实验 + 汇总 + 打印 OK =====
if ($SelfCheck) {
    $scOut = Join-Path $RunRoot "selfcheck_one"
    New-Item -ItemType Directory -Force -Path $scOut | Out-Null

    $args = @("--out_root", $scOut) + $baseArgs + @("--from_step","6","--to_step","7") + $CommonArgs
    $cmdStr = ($args | ForEach-Object { Quote-Arg $_ }) -join " "
    Write-Host "[RUN] $py $Pipeline $cmdStr"
    & $py $Pipeline @args

    if (Test-Path $SummaryScript) {
        $summaryPath = Join-Path $RunRoot "summary.xlsx"
        Write-Host "[RUN] $py $SummaryScript --root $RunRoot --out $summaryPath"
        & $py $SummaryScript --root $RunRoot --out $summaryPath

        $code = @"
import os, openpyxl as ox
p = r'''$summaryPath'''
assert os.path.exists(p), p
wb = ox.load_workbook(p)
assert len(wb.sheetnames) > 0, 'no sheets'
print('OK')
"@
        & $py -c $code
    } else {
        Write-Host "OK"
    }
    exit 0
}

# ===== 共享 Stage1（到 step9），产出 ckpt + thermal_UD 等一次性产物 =====
$sharedOut = Join-Path $RunRoot $SharedName
$sharedRgbModel = Join-Path $sharedOut "Model_RGB"
$sharedCkpt = Join-Path $sharedRgbModel ("chkpnt{0}.pth" -f $RgbIter)

if ($ReuseStage1) {
    if (-not (Test-Path $sharedCkpt)) {
        New-Item -ItemType Directory -Force -Path $sharedOut | Out-Null
        $sharedArgs = @("--out_root", $sharedOut) + $baseArgs + @("--from_step","1","--to_step","9") + $CommonArgs
        $cmdStr = ($sharedArgs | ForEach-Object { Quote-Arg $_ }) -join " "
        Write-Host "[RUN][SHARED] $py $Pipeline $cmdStr"
        & $py $Pipeline @sharedArgs
    }
    if (-not (Test-Path $sharedCkpt)) {
        throw "Shared RGB checkpoint not found after shared run: $sharedCkpt"
    }
    Write-Host "[INFO] Reusing shared RGB checkpoint: $sharedCkpt"
}

# ===== Stage2 实验列表（默认只做 stage2 相关消融；stage1 相关如 ss_enable_rgb/ss_enable 不放这里）=====
$experiments = @(
    @{ Name = "sgf_on";        Args = @() },
    @{ Name = "sgf_off";       Args = @("--sgf_disable") },

    @{ Name = "ss_aabb_t";     Args = @("--ss_enable_t", "--ss_source", "colmap_sparse", "--ss_aabb_margin", "0.0") },
    @{ Name = "ss_nn_t";       Args = @("--ss_enable_t", "--ss_source", "colmap_sparse", "--ss_aabb_margin", "0.0", "--ss_voxel_size", "0.5", "--ss_nn_dist_thr", "0.8") },

    @{ Name = "clamp3";        Args = @("--clamp_scale_max", "3") },
    @{ Name = "thermal_reset"; Args = @("--thermal_reset_features") },

    @{ Name = "debug_stats";   Args = @("--debug_gaussian_stats") },

    @{ Name = "ours_full";     Args = @("--ss_enable_t", "--ss_source", "colmap_sparse", "--ss_aabb_margin", "0.0", "--ss_voxel_size", "0.5", "--ss_nn_dist_thr", "0.8",
                                        "--clamp_scale_max", "3", "--thermal_reset_features") }
)

if ($Only.Count -gt 0) {
    $experiments = $experiments | Where-Object { $Only -contains $_.Name }
    if ($experiments.Count -eq 0) { throw "No experiments matched -Only. Given: $($Only -join ', ')" }
}

$results = @()
foreach ($exp in $experiments) {
    $outDir = Join-Path $RunRoot $exp.Name
    New-Item -ItemType Directory -Force -Path $outDir | Out-Null

    # 复用共享 Model_RGB：为每个实验目录创建 Model_RGB junction
    if ($ReuseStage1) {
        $linkPath = Join-Path $outDir "Model_RGB"
        Ensure-Junction $linkPath $sharedRgbModel
    }

    $expArgs = @()
    if ($ProfileMode -eq "audit" -and ($ProfileLogsFor -contains $exp.Name)) {
        $expArgs += @("--profile_save_logs")
    }

    $stepArgs = @("--from_step", $FromStep, "--to_step", $ToStep)

    $args = @("--out_root", $outDir) + $baseArgs + $exp.Args + $expArgs + $stepArgs + $CommonArgs

    $cmdStr = ($args | ForEach-Object { Quote-Arg $_ }) -join " "
    Write-Host "[RUN] $py $Pipeline $cmdStr"

    $status = "OK"
    try {
        & $py $Pipeline @args
    } catch {
        $status = "FAIL"
        Write-Warning "Experiment '$($exp.Name)' failed: $($_.Exception.Message)"
    }
    $results += [pscustomobject]@{ Name = $exp.Name; Status = $status; OutDir = $outDir }
}

# ===== 汇总 =====
$summaryPath = Join-Path $RunRoot "summary.xlsx"
if (Test-Path $SummaryScript) {
    Write-Host "[RUN] $py $SummaryScript --root $RunRoot --out $summaryPath"
    try {
        & $py $SummaryScript --root $RunRoot --out $summaryPath
    } catch {
        Write-Warning "summarize_ablation_excel.py failed: $($_.Exception.Message)"
    }
}

Write-Host "RUN_ROOT: $RunRoot"
Write-Host "SUMMARY: $summaryPath"

# 如果有失败，返回非 0（方便你在 CI/批处理里捕获）
if (($results | Where-Object { $_.Status -ne "OK" }).Count -gt 0) { exit 1 }
exit 0
