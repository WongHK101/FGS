$ErrorActionPreference = 'Continue'
$py = 'D:\anaconda\envs\fgs\python.exe'
$repo = 'd:\dataset\FGS\FGS-0202v1'
$nv = Join-Path $repo 'novel_view_metrics.py'
$source = 'E:\3DGS\probe_data\clean_ablation_rawRebuild_20260323\PVpanel'
$outRoot = 'D:\test1\pvpanel_full_grid72_r1_fix_20260328'
$logRoot = Join-Path $outRoot 'logs'
$flag = Join-Path $outRoot '_RUNNING.flag'

New-Item -ItemType Directory -Force -Path $outRoot | Out-Null
New-Item -ItemType Directory -Force -Path $logRoot | Out-Null
Set-Content -Path $flag -Value (Get-Date).ToString('yyyy-MM-dd HH:mm:ss') -Encoding ASCII

$runs = @(
  @{ group='M00_latest'; key='RGB'; sub='M00_latest\\PVpanel\\RGB'; model='F:\databackup\xr6\output\Ch4_2_MainComparison\M00_Baseline_ModulesOff_rawRebuild_20260323\PVpanel\Model_RGB' },
  @{ group='M00_latest'; key='T'; sub='M00_latest\\PVpanel\\T'; model='F:\databackup\xr6\output\Ch4_2_MainComparison\M00_Baseline_ModulesOff_rawRebuild_20260323\PVpanel\Model_T' },
  @{ group='M00_latest'; key='F_sh_opacity_geom_a0'; sub='M00_latest\\PVpanel\\F_sh_opacity_geom_a0'; model='F:\databackup\xr6\output\Ch4_2_MainComparison\M00_Baseline_ModulesOff_rawRebuild_20260323\PVpanel\Model_F\sh_opacity_geom\0' },
  @{ group='M00_latest'; key='F_sh_opacity_geom_a0p25'; sub='M00_latest\\PVpanel\\F_sh_opacity_geom_a0p25'; model='F:\databackup\xr6\output\Ch4_2_MainComparison\M00_Baseline_ModulesOff_rawRebuild_20260323\PVpanel\Model_F\sh_opacity_geom\0.25' },
  @{ group='M00_latest'; key='F_sh_opacity_geom_a0p5'; sub='M00_latest\\PVpanel\\F_sh_opacity_geom_a0p5'; model='F:\databackup\xr6\output\Ch4_2_MainComparison\M00_Baseline_ModulesOff_rawRebuild_20260323\PVpanel\Model_F\sh_opacity_geom\0.5' },
  @{ group='M00_latest'; key='F_sh_opacity_geom_a0p75'; sub='M00_latest\\PVpanel\\F_sh_opacity_geom_a0p75'; model='F:\databackup\xr6\output\Ch4_2_MainComparison\M00_Baseline_ModulesOff_rawRebuild_20260323\PVpanel\Model_F\sh_opacity_geom\0.75' },
  @{ group='M00_latest'; key='F_sh_opacity_geom_a1'; sub='M00_latest\\PVpanel\\F_sh_opacity_geom_a1'; model='F:\databackup\xr6\output\Ch4_2_MainComparison\M00_Baseline_ModulesOff_rawRebuild_20260323\PVpanel\Model_F\sh_opacity_geom\1' },

  @{ group='G01_latest'; key='RGB'; sub='G01_latest\\PVpanel\\RGB'; model='F:\databackup\xr6\output\Ch4_2_MainComparison\G01_M00_plus_SSP_rawRebuild_20260323\PVpanel\Model_RGB' },
  @{ group='G01_latest'; key='T'; sub='G01_latest\\PVpanel\\T'; model='F:\databackup\xr6\output\Ch4_2_MainComparison\G01_M00_plus_SSP_rawRebuild_20260323\PVpanel\Model_T' },
  @{ group='G01_latest'; key='F_sh_opacity_geom_a0'; sub='G01_latest\\PVpanel\\F_sh_opacity_geom_a0'; model='F:\databackup\xr6\output\Ch4_2_MainComparison\G01_M00_plus_SSP_rawRebuild_20260323\PVpanel\Model_F\sh_opacity_geom\0' },
  @{ group='G01_latest'; key='F_sh_opacity_geom_a0p25'; sub='G01_latest\\PVpanel\\F_sh_opacity_geom_a0p25'; model='F:\databackup\xr6\output\Ch4_2_MainComparison\G01_M00_plus_SSP_rawRebuild_20260323\PVpanel\Model_F\sh_opacity_geom\0.25' },
  @{ group='G01_latest'; key='F_sh_opacity_geom_a0p5'; sub='G01_latest\\PVpanel\\F_sh_opacity_geom_a0p5'; model='F:\databackup\xr6\output\Ch4_2_MainComparison\G01_M00_plus_SSP_rawRebuild_20260323\PVpanel\Model_F\sh_opacity_geom\0.5' },
  @{ group='G01_latest'; key='F_sh_opacity_geom_a0p75'; sub='G01_latest\\PVpanel\\F_sh_opacity_geom_a0p75'; model='F:\databackup\xr6\output\Ch4_2_MainComparison\G01_M00_plus_SSP_rawRebuild_20260323\PVpanel\Model_F\sh_opacity_geom\0.75' },
  @{ group='G01_latest'; key='F_sh_opacity_geom_a1'; sub='G01_latest\\PVpanel\\F_sh_opacity_geom_a1'; model='F:\databackup\xr6\output\Ch4_2_MainComparison\G01_M00_plus_SSP_rawRebuild_20260323\PVpanel\Model_F\sh_opacity_geom\1' },

  @{ group='G02_latest'; key='RGB'; sub='G02_latest\\PVpanel\\RGB'; model='F:\databackup\xr6\output\Ch4_2_MainComparison\G02_M00_plus_STT_rawRebuild_20260323\PVpanel\Model_RGB' },
  @{ group='G02_latest'; key='T'; sub='G02_latest\\PVpanel\\T'; model='F:\databackup\xr6\output\Ch4_2_MainComparison\G02_M00_plus_STT_rawRebuild_20260323\PVpanel\Model_T' },
  @{ group='G02_latest'; key='F_sh_opacity_geom_a0'; sub='G02_latest\\PVpanel\\F_sh_opacity_geom_a0'; model='F:\databackup\xr6\output\Ch4_2_MainComparison\G02_M00_plus_STT_rawRebuild_20260323\PVpanel\Model_F\sh_opacity_geom\0' },
  @{ group='G02_latest'; key='F_sh_opacity_geom_a0p25'; sub='G02_latest\\PVpanel\\F_sh_opacity_geom_a0p25'; model='F:\databackup\xr6\output\Ch4_2_MainComparison\G02_M00_plus_STT_rawRebuild_20260323\PVpanel\Model_F\sh_opacity_geom\0.25' },
  @{ group='G02_latest'; key='F_sh_opacity_geom_a0p5'; sub='G02_latest\\PVpanel\\F_sh_opacity_geom_a0p5'; model='F:\databackup\xr6\output\Ch4_2_MainComparison\G02_M00_plus_STT_rawRebuild_20260323\PVpanel\Model_F\sh_opacity_geom\0.5' },
  @{ group='G02_latest'; key='F_sh_opacity_geom_a0p75'; sub='G02_latest\\PVpanel\\F_sh_opacity_geom_a0p75'; model='F:\databackup\xr6\output\Ch4_2_MainComparison\G02_M00_plus_STT_rawRebuild_20260323\PVpanel\Model_F\sh_opacity_geom\0.75' },
  @{ group='G02_latest'; key='F_sh_opacity_geom_a1'; sub='G02_latest\\PVpanel\\F_sh_opacity_geom_a1'; model='F:\databackup\xr6\output\Ch4_2_MainComparison\G02_M00_plus_STT_rawRebuild_20260323\PVpanel\Model_F\sh_opacity_geom\1' },

  @{ group='M01_latest'; key='RGB'; sub='M01_latest\\PVpanel\\RGB'; model='F:\databackup\xr6\output\Ch4_2_MainComparison\M01_OursFull_Default_rawRebuild_step1to14_20260325\PVpanel\Model_RGB' },
  @{ group='M01_latest'; key='T'; sub='M01_latest\\PVpanel\\T'; model='F:\databackup\xr6\output\Ch4_2_MainComparison\M01_OursFull_Default_rawRebuild_step1to14_20260325\PVpanel\Model_T' },
  @{ group='M01_latest'; key='F_sh_opacity_geom_a0'; sub='M01_latest\\PVpanel\\F_sh_opacity_geom_a0'; model='F:\databackup\xr6\output\Ch4_2_MainComparison\M01_OursFull_Default_rawRebuild_step1to14_20260325\PVpanel\Model_F\sh_opacity_geom\0' },
  @{ group='M01_latest'; key='F_sh_opacity_geom_a0p25'; sub='M01_latest\\PVpanel\\F_sh_opacity_geom_a0p25'; model='F:\databackup\xr6\output\Ch4_2_MainComparison\M01_OursFull_Default_rawRebuild_step1to14_20260325\PVpanel\Model_F\sh_opacity_geom\0.25' },
  @{ group='M01_latest'; key='F_sh_opacity_geom_a0p5'; sub='M01_latest\\PVpanel\\F_sh_opacity_geom_a0p5'; model='F:\databackup\xr6\output\Ch4_2_MainComparison\M01_OursFull_Default_rawRebuild_step1to14_20260325\PVpanel\Model_F\sh_opacity_geom\0.5' },
  @{ group='M01_latest'; key='F_sh_opacity_geom_a0p75'; sub='M01_latest\\PVpanel\\F_sh_opacity_geom_a0p75'; model='F:\databackup\xr6\output\Ch4_2_MainComparison\M01_OursFull_Default_rawRebuild_step1to14_20260325\PVpanel\Model_F\sh_opacity_geom\0.75' },
  @{ group='M01_latest'; key='F_sh_opacity_geom_a1'; sub='M01_latest\\PVpanel\\F_sh_opacity_geom_a1'; model='F:\databackup\xr6\output\Ch4_2_MainComparison\M01_OursFull_Default_rawRebuild_step1to14_20260325\PVpanel\Model_F\sh_opacity_geom\1' }
)

$rows = New-Object System.Collections.Generic.List[Object]
$idx = 0
$total = $runs.Count
foreach ($r in $runs) {
  $idx += 1
  $outDir = Join-Path $outRoot $r.sub
  New-Item -ItemType Directory -Force -Path $outDir | Out-Null
  $logPath = Join-Path $logRoot ("{0:00}_{1}_{2}.log" -f $idx, $r.group, $r.key)
  $start = Get-Date
  $args = @($nv, '--model_path', $r.model, '--source_path', $source, '--mode', 'grid72', '-r', '1', '--out_dir', $outDir)
  & $py @args *> $logPath
  $rc = $LASTEXITCODE
  $elapsed = [math]::Round(((Get-Date) - $start).TotalSeconds, 1)
  $count = (Get-ChildItem -Path $outDir -Filter '*.png' -File -ErrorAction SilentlyContinue | Measure-Object).Count
  $rows.Add([pscustomobject]@{
      index = $idx
      total = $total
      group = $r.group
      key = $r.key
      returncode = $rc
      render_count = $count
      elapsed_sec = $elapsed
      out_dir = $outDir
      log = $logPath
      model_path = $r.model
  }) | Out-Null
  $rows | Export-Csv -Path (Join-Path $outRoot 'render_manifest.csv') -NoTypeInformation -Encoding UTF8
}

if (Test-Path $flag) { Remove-Item -Path $flag -Force }
