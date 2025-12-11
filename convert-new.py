import os
import logging
from argparse import ArgumentParser
import shutil


def q(p: str) -> str:
    """Quote a path for command line usage."""
    return f"\"{p}\""


def _pick_best_model_dir(base_dir: str):
    """
    Choose the best model directory inside base_dir by images.{bin,txt} size,
    as a proxy for number of registered images.
    """
    if not os.path.isdir(base_dir):
        return None
    best_dir = None
    best_size = -1
    for d in os.listdir(base_dir):
        full = os.path.join(base_dir, d)
        if not os.path.isdir(full):
            continue
        cand_bin = os.path.join(full, "images.bin")
        cand_txt = os.path.join(full, "images.txt")
        if os.path.isfile(cand_bin):
            size = os.path.getsize(cand_bin)
        elif os.path.isfile(cand_txt):
            size = os.path.getsize(cand_txt)
        else:
            size = -1
        if size > best_size:
            best_size = size
            best_dir = full
    return best_dir


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
    best_model_dir = _pick_best_model_dir(dist_sparse)
    if best_model_dir is None:
        # 回退到 sparse/0
        best_model_dir = os.path.join(dist_sparse, "0")

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
