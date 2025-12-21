import os
import logging
from argparse import ArgumentParser
import shutil
import subprocess
import re


def q(p: str) -> str:
    """Quote a path for command line usage."""
    return f"\"{p}\""



def _read_uint64(path: str):
    try:
        with open(path, "rb") as f:
            data = f.read(8)
            if len(data) != 8:
                return None
            import struct
            return int(struct.unpack("<Q", data)[0])
    except Exception:
        return None


def _count_registered_images(model_dir: str):
    """Return number of registered images in a COLMAP model folder, or None on failure."""
    # Binary format: first uint64 is number of registered images
    img_bin = os.path.join(model_dir, "images.bin")
    if os.path.isfile(img_bin):
        return _read_uint64(img_bin)

    # TXT fallback (images.txt): count non-comment image lines (every 2 lines per image in COLMAP TXT)
    img_txt = os.path.join(model_dir, "images.txt")
    if os.path.isfile(img_txt):
        try:
            n = 0
            with open(img_txt, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    # In images.txt, each image has a header line starting with image_id and a following 2D points line.
                    # We count header lines by checking whether the first token is an integer.
                    tok = line.split()
                    if tok and tok[0].isdigit():
                        n += 1
            return n if n > 0 else None
        except Exception:
            return None

    return None


def _count_points3d(model_dir: str):
    """Return number of 3D points in a COLMAP model folder, or None on failure."""
    pts_bin = os.path.join(model_dir, "points3D.bin")
    if os.path.isfile(pts_bin):
        return _read_uint64(pts_bin)

    pts_txt = os.path.join(model_dir, "points3D.txt")
    if os.path.isfile(pts_txt):
        try:
            n = 0
            with open(pts_txt, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    tok = line.split()
                    if tok and tok[0].isdigit():
                        n += 1
            return n if n > 0 else None
        except Exception:
            return None

    return None


def _pick_best_model_dir(base_dir: str, colmap_exec: str = "colmap"):
    """
    Choose the best model directory inside base_dir.

    Priority:
      1) Max registered images (from images.bin/images.txt)
      2) Tie-breaker: Max Points3D (from points3D.bin/points3D.txt)

    Tie-breakers / fallbacks:
      A) Largest points3D.bin
      B) Largest images.{bin,txt} (historical proxy)
    """
    if not os.path.isdir(base_dir):
        return None

    candidates = []
    for d in os.listdir(base_dir):
        full = os.path.join(base_dir, d)
        if not os.path.isdir(full):
            continue
        # A valid COLMAP model directory should have cameras & images at minimum.
        if not (
            os.path.isfile(os.path.join(full, "cameras.bin"))
            and os.path.isfile(os.path.join(full, "images.bin"))
        ):
            continue

        reg = _count_registered_images(full)
        pts = _count_points3d(full)
        pts_bin = os.path.join(full, "points3D.bin")
        pts_size = os.path.getsize(pts_bin) if os.path.isfile(pts_bin) else -1
        img_bin = os.path.join(full, "images.bin")
        img_txt = os.path.join(full, "images.txt")
        img_size = os.path.getsize(img_bin) if os.path.isfile(img_bin) else (os.path.getsize(img_txt) if os.path.isfile(img_txt) else -1)

        candidates.append(
            {
                "dir": full,
                "name": d,
                "reg": reg,
                "pts": pts,
                "pts_size": pts_size,
                "img_size": img_size,
            }
        )

    if not candidates:
        return None

    # Sort by registered images, then points3D, then file sizes (as tie-breakers).
    candidates.sort(
        key=lambda c: (
            c["reg"] if c["reg"] is not None else -1,
            c["pts"] if c["pts"] is not None else -1,
            c["pts_size"],
            c["img_size"],
        ),
        reverse=True,
    )


    # Log top few for transparency.
    logging.info("Sparse models found under %s:", base_dir)
    for c in candidates[:10]:
        logging.info(
            "  model=%s | registered=%s | points3D=%s | points3D.bin=%s bytes | images=%s bytes",
            c["name"],
            "?" if c["reg"] is None else c["reg"],
            "?" if c["pts"] is None else c["pts"],
            c["pts_size"],
            c["img_size"],
        )

    return candidates[0]["dir"]


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = ArgumentParser("Colmap converter (extended)")

    # 原有参数
    parser.add_argument("--no_gpu", action="store_true")
    parser.add_argument("--skip_matching", action="store_true")
    parser.add_argument("--source_path", "-s", required=True, type=str)
    parser.add_argument("--camera", default="OPENCV", type=str)
    parser.add_argument("--colmap_executable", default="", type=str)
    parser.add_argument("--resize", action="store_true")
    parser.add_argument("--magick_executable", default="", type=str)

    # 匹配模式：支持 exhaustive / sequential / spatial / vocab_tree
    parser.add_argument(
        "--matching",
        type=str,
        default="exhaustive",
        choices=["exhaustive", "sequential", "spatial", "vocab_tree"],
        help="Which COLMAP matcher to use (*_matcher). Default: exhaustive_matcher",
    )

    # Mapper 相关便捷参数（仅在提供时转发）
    parser.add_argument("--min_model_size", type=int, default=None)           # -> Mapper.min_model_size
    parser.add_argument("--mapper_multiple_models", type=int, default=None)   # -> Mapper.multiple_models
    parser.add_argument("--init_min_num_inliers", type=int, default=None)     # -> Mapper.init_min_num_inliers
    parser.add_argument("--abs_pose_min_num_inliers", type=int, default=None) # -> Mapper.abs_pose_min_num_inliers
    parser.add_argument(
        "--mapper_extra",
        type=str,
        default="",
        help="Extra mapper flags (kept for backward compatibility).",
    )

    # 通用透传参数：支持原版 COLMAP 的所有参数
    parser.add_argument(
        "--feature_extractor_args",
        type=str,
        default="",
        help="Extra arguments appended to `colmap feature_extractor`.",
    )
    parser.add_argument(
        "--matcher_args",
        type=str,
        default="",
        help="Extra arguments appended to the matcher (exhaustive/sequential/spatial/vocab_tree).",
    )
    parser.add_argument(
        "--mapper_args",
        type=str,
        default="",
        help="Extra arguments appended to `colmap mapper`.",
    )
    parser.add_argument(
        "--image_undistorter_args",
        type=str,
        default="",
        help="Extra arguments appended to `colmap image_undistorter`.",
    )

    # model_aligner 开关 + 透传参数
    parser.add_argument(
        "--use_model_aligner",
        action="store_true",
        help="If set, run `colmap model_aligner` after mapper and before image_undistorter.",
    )
    parser.add_argument(
        "--model_aligner_args",
        type=str,
        default="",
        help="Extra arguments appended to `colmap model_aligner` "
             "(e.g. --ref_is_gps=1 --alignment_type=enu --robust_alignment=1 ...).",
    )

    args = parser.parse_args()

    # 可执行文件
    # - colmap_command: for os.system (string, may be quoted)
    # - colmap_exec: for subprocess (argv list, must be unquoted)
    colmap_exec = args.colmap_executable.strip('"') if len(args.colmap_executable) > 0 else "colmap"
    colmap_command = f"\"{args.colmap_executable}\"" if len(args.colmap_executable) > 0 else "colmap"
    magick_command = f"\"{args.magick_executable}\"" if len(args.magick_executable) > 0 else "magick"
    use_gpu = 1 if not args.no_gpu else 0

    # 路径
    src = args.source_path
    distorted_dir = os.path.join(src, "distorted")
    os.makedirs(distorted_dir, exist_ok=True)

    dist_db = os.path.join(distorted_dir, "database.db")
    input_dir = os.path.join(src, "input")
    dist_sparse = os.path.join(distorted_dir, "sparse")

    # -------------------------------------------------------------------------
    # 1) COLMAP: feature_extractor + matcher + mapper
    # -------------------------------------------------------------------------
    if not args.skip_matching:
        os.makedirs(dist_sparse, exist_ok=True)

        # 1.1 Feature extraction
        feat_extraction_cmd = (
            f"{colmap_command} feature_extractor "
            f"--database_path {q(dist_db)} "
            f"--image_path {q(input_dir)} "
            f"--ImageReader.single_camera 1 "
            f"--ImageReader.camera_model {args.camera} "
            f"--SiftExtraction.use_gpu {use_gpu}"
        )
        if args.feature_extractor_args:
            feat_extraction_cmd += " " + args.feature_extractor_args

        logging.info("Running feature_extractor:")
        logging.info(feat_extraction_cmd)
        exit_code = os.system(feat_extraction_cmd)
        if exit_code != 0:
            logging.error(f"Feature extraction failed with code {exit_code}. Exiting.")
            raise SystemExit(exit_code)

        # 1.2 Matching (exhaustive / sequential / spatial / vocab_tree)
        matcher_tool = {
            "exhaustive": "exhaustive_matcher",
            "sequential": "sequential_matcher",
            "spatial": "spatial_matcher",
            "vocab_tree": "vocab_tree_matcher",
        }[args.matching]

        feat_matching_cmd = (
            f"{colmap_command} {matcher_tool} "
            f"--database_path {q(dist_db)} "
            f"--SiftMatching.use_gpu {use_gpu}"
        )
        if args.matcher_args:
            feat_matching_cmd += " " + args.matcher_args

        logging.info("Running matcher:")
        logging.info(feat_matching_cmd)
        exit_code = os.system(feat_matching_cmd)
        if exit_code != 0:
            logging.error(f"Feature matching failed with code {exit_code}. Exiting.")
            raise SystemExit(exit_code)

        # 1.3 Mapping / bundle adjustment
        mapper_parts = [
            f"{colmap_command} mapper",
            f"--database_path {q(dist_db)}",
            f"--image_path {q(input_dir)}",
            f"--output_path {q(dist_sparse)}",
            "--Mapper.ba_global_function_tolerance=0.00001",
        ]
        if args.min_model_size is not None:
            mapper_parts.append(f"--Mapper.min_model_size={args.min_model_size}")
        if args.mapper_multiple_models is not None:
            mapper_parts.append(f"--Mapper.multiple_models={args.mapper_multiple_models}")
        if args.init_min_num_inliers is not None:
            mapper_parts.append(f"--Mapper.init_min_num_inliers={args.init_min_num_inliers}")
        if args.abs_pose_min_num_inliers is not None:
            mapper_parts.append(f"--Mapper.abs_pose_min_num_inliers={args.abs_pose_min_num_inliers}")
        if args.mapper_extra:
            mapper_parts.append(args.mapper_extra)
        if args.mapper_args:
            mapper_parts.append(args.mapper_args)

        mapper_cmd = " ".join(mapper_parts)
        logging.info("Running mapper:")
        logging.info(mapper_cmd)
        exit_code = os.system(mapper_cmd)
        if exit_code != 0:
            logging.error(f"Mapper failed with code {exit_code}. Exiting.")
            raise SystemExit(exit_code)

    # -------------------------------------------------------------------------
    # 2) 选择最佳 sparse 模型（可能有多个子模型）
    # -------------------------------------------------------------------------
    best_model_dir = _pick_best_model_dir(dist_sparse, colmap_exec=colmap_exec)
    if best_model_dir is None:
        # 回退到 sparse/0
        best_model_dir = os.path.join(dist_sparse, "0")

    logging.info("Selected best sparse model: %s", best_model_dir)

    if not os.path.isdir(best_model_dir):
        logging.error(f"No valid sparse model found under {dist_sparse}. "
                      f"Expected something like {os.path.join(dist_sparse, '0')}.")
        raise SystemExit(1)

    # -------------------------------------------------------------------------
    # 3) 可选：model_aligner（如需对齐到 GPS / ENU / ECEF 等坐标系）
    # -------------------------------------------------------------------------
    if args.use_model_aligner:
        aligned_sparse = os.path.join(distorted_dir, "sparse_aligned")
        os.makedirs(aligned_sparse, exist_ok=True)

        model_aligner_cmd = (
            f"{colmap_command} model_aligner "
            f"--input_path {q(best_model_dir)} "
            f"--output_path {q(aligned_sparse)} "
            f"--database_path {q(dist_db)}"
        )
        if args.model_aligner_args:
            model_aligner_cmd += " " + args.model_aligner_args

        logging.info("Running model_aligner:")
        logging.info(model_aligner_cmd)
        exit_code = os.system(model_aligner_cmd)
        if exit_code != 0:
            logging.error(f"model_aligner failed with code {exit_code}. Exiting.")
            raise SystemExit(exit_code)

        # 对齐后的模型作为 image_undistorter 的输入
        best_model_dir = aligned_sparse

    # -------------------------------------------------------------------------
    # 4) Image undistortion（生成 3DGS 所需的 images + sparse/0）
    # -------------------------------------------------------------------------
    img_undist_cmd = (
        f"{colmap_command} image_undistorter "
        f"--image_path {q(input_dir)} "
        f"--input_path {q(best_model_dir)} "
        f"--output_path {q(src)} "
        f"--output_type COLMAP"
    )
    if args.image_undistorter_args:
        img_undist_cmd += " " + args.image_undistorter_args

    logging.info("Running image_undistorter:")
    logging.info(img_undist_cmd)
    exit_code = os.system(img_undist_cmd)
    if exit_code != 0:
        logging.error(f"Image undistorter failed with code {exit_code}. Exiting.")
        raise SystemExit(exit_code)

    # 将 undistorted sparse 文件移动到 sparse/0
    sparse_dir = os.path.join(src, "sparse")
    if os.path.isdir(sparse_dir):
        files = os.listdir(sparse_dir)
        target_dir = os.path.join(sparse_dir, "0")
        os.makedirs(target_dir, exist_ok=True)
        for file in files:
            if file == "0":
                continue
            source_file = os.path.join(sparse_dir, file)
            destination_file = os.path.join(target_dir, file)
            if os.path.isfile(source_file):
                shutil.move(source_file, destination_file)

    # -------------------------------------------------------------------------
    # 5) 可选：多分辨率缩放（ImageMagick）
    # -------------------------------------------------------------------------
    if args.resize:
        logging.info("Copying and resizing images (ImageMagick)...")

        images_dir = os.path.join(src, "images")
        images_2 = os.path.join(src, "images_2")
        images_4 = os.path.join(src, "images_4")
        images_8 = os.path.join(src, "images_8")
        os.makedirs(images_2, exist_ok=True)
        os.makedirs(images_4, exist_ok=True)
        os.makedirs(images_8, exist_ok=True)

        if os.path.isdir(images_dir):
            files = os.listdir(images_dir)
            for file in files:
                source_file = os.path.join(images_dir, file)

                # 1/2 尺度
                destination_file = os.path.join(images_2, file)
                shutil.copy2(source_file, destination_file)
                cmd_50 = f"{magick_command} mogrify -resize 50% {q(destination_file)}"
                exit_code = os.system(cmd_50)
                if exit_code != 0:
                    logging.error(f"50% resize failed with code {exit_code}. Exiting.")
                    raise SystemExit(exit_code)

                # 1/4 尺度
                destination_file = os.path.join(images_4, file)
                shutil.copy2(source_file, destination_file)
                cmd_25 = f"{magick_command} mogrify -resize 25% {q(destination_file)}"
                exit_code = os.system(cmd_25)
                if exit_code != 0:
                    logging.error(f"25% resize failed with code {exit_code}. Exiting.")
                    raise SystemExit(exit_code)

                # 1/8 尺度
                destination_file = os.path.join(images_8, file)
                shutil.copy2(source_file, destination_file)
                cmd_125 = f"{magick_command} mogrify -resize 12.5% {q(destination_file)}"
                exit_code = os.system(cmd_125)
                if exit_code != 0:
                    logging.error(f"12.5% resize failed with code {exit_code}. Exiting.")
                    raise SystemExit(exit_code)

    logging.info("Done.")


if __name__ == "__main__":
    main()
