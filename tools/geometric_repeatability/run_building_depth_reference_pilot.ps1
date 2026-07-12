param(
    [string]$RepoRoot = "D:\dataset\FGS\FGS-0202v1",
    [string]$PythonExe = "D:\anaconda\envs\fgs\python.exe",
    [string]$StrictProtocolManifest = "G:\ACMMM26\GeometricRepeatability\Building\M01_Strict_v1\strict_dataset\strict_protocol_manifest.json",
    [string]$OutRoot = "G:\ACMMM26\GeometricRepeatability\DepthReference\Building_even5_pilot_v1",
    [string]$ColmapCmd = "C:\Program Files\colmap\COLMAP.bat",
    [switch]$EnableReferenceMeshHoleFix
)

$ErrorActionPreference = "Stop"

function Assert-Exists {
    param(
        [Parameter(Mandatory = $true)][string]$PathValue,
        [Parameter(Mandatory = $true)][string]$Label
    )
    if (-not (Test-Path -LiteralPath $PathValue)) {
        throw "$Label not found: $PathValue"
    }
}

function Write-JsonUtf8 {
    param(
        [Parameter(Mandatory = $true)]$Object,
        [Parameter(Mandatory = $true)][string]$PathValue
    )
    $dir = Split-Path -Parent $PathValue
    if ($dir) {
        New-Item -ItemType Directory -Force -Path $dir | Out-Null
    }
    $Object | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $PathValue -Encoding UTF8
}

Assert-Exists -PathValue $RepoRoot -Label "Repo root"
Assert-Exists -PathValue $PythonExe -Label "FGS python"
Assert-Exists -PathValue $StrictProtocolManifest -Label "Strict protocol manifest"
Assert-Exists -PathValue $ColmapCmd -Label "COLMAP command"

New-Item -ItemType Directory -Force -Path $OutRoot | Out-Null
$TranscriptPath = Join-Path $OutRoot "run_transcript.txt"
Start-Transcript -Path $TranscriptPath -Append -Force | Out-Null

$status = [ordered]@{
    stage = "starting"
    started_at = (Get-Date).ToString("s")
    out_root = $OutRoot
    strict_protocol_manifest = $StrictProtocolManifest
    enable_reference_mesh_hole_fix = [bool]$EnableReferenceMeshHoleFix
    transcript = $TranscriptPath
}
$StatusJson = Join-Path $OutRoot "status.json"
Write-JsonUtf8 -Object $status -PathValue $StatusJson

try {
    $strict = Get-Content -LiteralPath $StrictProtocolManifest -Raw | ConvertFrom-Json
    $lists = $strict.lists
    $trainEven = [string](Join-Path $strict.split_dir "train_even.txt")
    $probeList = [string]$lists.probe_test
    $strictThermalRoot = [string]$strict.artifacts.strict_thermal_root
    Assert-Exists -PathValue $trainEven -Label "train_even list"
    Assert-Exists -PathValue $probeList -Label "probe_test list"
    Assert-Exists -PathValue $strictThermalRoot -Label "strict thermal root"

    $referenceRoot = Join-Path $OutRoot "reference"
    $status.stage = "build_reference"
    Write-JsonUtf8 -Object $status -PathValue $StatusJson
    $referenceArgs = @(
        (Join-Path $RepoRoot "tools\geometric_repeatability\build_depth_reference.py"),
        "--strict_protocol_manifest", $StrictProtocolManifest,
        "--out_dir", $referenceRoot,
        "--colmap_cmd", $ColmapCmd,
        "--resolution_arg", "4",
        "--thresholds_m", "0.10,0.25,0.50,1.00,2.00,5.00,10.00,20.00,30.00",
        "--support_min_count", "1",
        "--support_radius_px", "1",
        "--support_depth_tolerance_m", "0.10"
    )
    if ($EnableReferenceMeshHoleFix) {
        $referenceArgs += @(
            "--mesh_backend_preference", "poisson",
            "--poisson_depth", "13",
            "--poisson_trim", "7",
            "--poisson_point_weight", "4",
            "--stereo_fusion_min_num_pixels", "2",
            "--stereo_fusion_max_depth_error", "0.05",
            "--stereo_fusion_max_reproj_error", "4.0",
            "--stereo_fusion_max_normal_error", "20",
            "--patch_match_window_radius", "7",
            "--patch_match_num_iterations", "7",
            "--patch_match_geom_consistency", "1",
            "--patch_match_filter", "1",
            "--patch_match_min_triangulation_angle", "0.3",
            "--patch_match_filter_min_triangulation_angle", "0.5",
            "--patch_match_filter_min_num_consistent", "1",
            "--patch_match_filter_min_ncc", "0.05",
            "--patch_match_filter_geom_consistency_max_cost", "2.0"
        )
    }
    & $PythonExe @referenceArgs
    if ($LASTEXITCODE -ne 0) { throw "Reference build failed" }

    $referenceManifest = Join-Path $referenceRoot "reference_depth_manifest.json"
    Assert-Exists -PathValue $referenceManifest -Label "reference depth manifest"

    $methods = @(
        [ordered]@{
            method_name = "Ours_M01_even"
            model_path = "G:\ACMMM26\GeometricRepeatability\Building\M01_Strict_v1\even\Model_T"
            iteration = 60000
            train_list = $trainEven
            source_path = $strictThermalRoot
            images = "images"
        },
        [ordered]@{
            method_name = "ThermalGaussian_OMMG_even"
            model_path = "G:\ACMMM26\GeometricRepeatability\CrossMethod\ThermalGaussian_OMMG\Building\Strict_v1\even"
            iteration = 30000
            train_list = $trainEven
            source_path = $strictThermalRoot
            images = "images"
        },
        [ordered]@{
            method_name = "ThermalGaussian_MFTG_even"
            model_path = "G:\ACMMM26\GeometricRepeatability\CrossMethod\ThermalGaussian_MFTG\Building\Strict_v1\even"
            iteration = 30000
            train_list = $trainEven
            source_path = $strictThermalRoot
            images = "images"
        },
        [ordered]@{
            method_name = "ThermalGaussian_MSMG_even"
            model_path = "G:\ACMMM26\GeometricRepeatability\CrossMethod\ThermalGaussian_MSMG\Building\Strict_v1\even"
            iteration = 30000
            train_list = $trainEven
            source_path = $strictThermalRoot
            images = "images"
        },
        [ordered]@{
            method_name = "Thermal3D_GS_even"
            model_path = "G:\ACMMM26\GeometricRepeatability\CrossMethod\Thermal3D_GS\Building\Strict_v1\even"
            iteration = 30000
            train_list = $trainEven
            source_path = $strictThermalRoot
            images = "images"
        }
    )

    $metricJsons = @()
    foreach ($method in $methods) {
        $methodName = [string]$method.method_name
        $methodRoot = Join-Path $OutRoot $methodName
        $bundleRoot = Join-Path $methodRoot "bundle"
        $evalRoot = Join-Path $methodRoot "evaluation"
        $adapterManifest = Join-Path $methodRoot "depth_adapter_manifest.json"
        New-Item -ItemType Directory -Force -Path $methodRoot | Out-Null

        Assert-Exists -PathValue ([string]$method.model_path) -Label "$methodName model path"
        $status.stage = "export_$methodName"
        Write-JsonUtf8 -Object $status -PathValue $StatusJson
        if (-not (Test-Path -LiteralPath (Join-Path $bundleRoot "split_manifest.json"))) {
            & $PythonExe (Join-Path $RepoRoot "tools\geometric_repeatability\export_gaussian_probe_bundle.py") `
                --model_path ([string]$method.model_path) `
                --source_path ([string]$method.source_path) `
                --images ([string]$method.images) `
                --train_list ([string]$method.train_list) `
                --test_list $probeList `
                --eval `
                --iteration ([int]$method.iteration) `
                --split_label "heldout_probe" `
                --scene_name_override "Building" `
                --out_dir $bundleRoot `
                --quiet
            if ($LASTEXITCODE -ne 0) { throw "$methodName bundle export failed" }
        }

        $adapter = [ordered]@{
            protocol_name = "reference-depth-based-geometric-evaluation-v1"
            method_name = $methodName
            model_path = [string]$method.model_path
            source_path = [string]$method.source_path
            iteration = [int]$method.iteration
            depth_semantics = "inverse_camera_z_from_renderer"
            validity_rule = [ordered]@{
                mode = "opacity_threshold"
                opacity_threshold = 0.5
                depth_min = 1e-6
            }
            notes = "Frozen v1 adapter for Gaussian renderer exports from export_gaussian_probe_bundle.py"
        }
        Write-JsonUtf8 -Object $adapter -PathValue $adapterManifest

        $status.stage = "evaluate_$methodName"
        Write-JsonUtf8 -Object $status -PathValue $StatusJson
        & $PythonExe (Join-Path $RepoRoot "tools\geometric_repeatability\evaluate_depth_reference.py") `
            --reference_manifest $referenceManifest `
            --model_manifest (Join-Path $bundleRoot "split_manifest.json") `
            --adapter_manifest $adapterManifest `
            --out_dir $evalRoot
        if ($LASTEXITCODE -ne 0) { throw "$methodName evaluation failed" }
        $metricJsons += (Join-Path $evalRoot "metrics_summary.json")
    }

    $summaryRoot = Join-Path $OutRoot "summary"
    $status.stage = "summarize"
    Write-JsonUtf8 -Object $status -PathValue $StatusJson
    & $PythonExe (Join-Path $RepoRoot "tools\geometric_repeatability\summarize_depth_reference_methods.py") `
        --metrics_json $metricJsons `
        --out_dir $summaryRoot `
        --rank_threshold_m 0.25
    if ($LASTEXITCODE -ne 0) { throw "Depth-reference summary failed" }

    $status.stage = "completed"
    $status.completed_at = (Get-Date).ToString("s")
    $status.reference_manifest = $referenceManifest
    $status.summary_root = $summaryRoot
    $status.metrics_jsons = $metricJsons
    Write-JsonUtf8 -Object $status -PathValue $StatusJson
}
catch {
    $status.stage = "failed"
    $status.failed_at = (Get-Date).ToString("s")
    $status.error = $_.Exception.Message
    Write-JsonUtf8 -Object $status -PathValue $StatusJson
    throw
}
finally {
    Stop-Transcript | Out-Null
}
