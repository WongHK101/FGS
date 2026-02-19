from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple, Optional

IMG_EXTS_DEFAULT = ["jpg", "jpeg", "tif", "tiff", "png", "heic", "dng"]


def iter_images(root: Path, recursive: bool, exts: List[str]) -> List[Path]:
    exts_lc = {e.lower().lstrip(".") for e in exts}
    files = [p for p in (root.rglob("*") if recursive else root.iterdir()) if p.is_file()]
    out = [p for p in files if p.suffix.lower().lstrip(".") in exts_lc]
    out.sort()
    return out


def make_key(p: Path, base_root: Path, match: str, case_insensitive: bool) -> str:
    if match == "basename":
        k = p.name
    elif match == "relative":
        k = str(p.relative_to(base_root)).replace("\\", "/")
    else:
        raise ValueError(match)
    return k.lower() if case_insensitive else k


def build_index(
    root: Path, recursive: bool, exts: List[str], match: str, case_insensitive: bool
) -> Tuple[Dict[str, Path], Dict[str, List[Path]]]:
    buckets: Dict[str, List[Path]] = {}
    for p in iter_images(root, recursive=recursive, exts=exts):
        k = make_key(p, root, match, case_insensitive)
        buckets.setdefault(k, []).append(p)

    unique: Dict[str, Path] = {}
    dups: Dict[str, List[Path]] = {}
    for k, lst in buckets.items():
        if len(lst) == 1:
            unique[k] = lst[0]
        else:
            dups[k] = lst
    return unique, dups


def chunk_list(items: List, n: int) -> List[List]:
    if n <= 0:
        return [items]
    return [items[i:i + n] for i in range(0, len(items), n)]


def _norm_path(p: Path) -> str:
    s = os.path.abspath(str(p))
    s = os.path.normpath(s)
    if os.name == "nt":
        s = os.path.normcase(s)
    return s


def run_exiftool_stay_open(
    exiftool: str,
    pairs: List[Tuple[Path, Path]],
    mode: str,
    clear_dst_gps: bool,
    keep_backup: bool,
    dry_run: bool,
) -> None:
    if not pairs:
        return

    with tempfile.NamedTemporaryFile("w", suffix=".args", delete=False, encoding="utf-8") as f:
        arg_path = f.name

        f.write("-common_args\n")
        f.write("-charset\nfilename=utf8\n")
        f.write("-m\n")
        f.write("-P\n")
        f.write("-q\n-q\n")
        if not keep_backup:
            # 不生成 *_original 备份
            f.write("-overwrite_original\n")

        for src, dst in pairs:
            if mode == "gps":
                if clear_dst_gps:
                    f.write("-GPS:all=\n")
                    f.write("-XMP-drone-dji:all=\n")
                f.write("-TagsFromFile\n")
                f.write(str(src) + "\n")
                f.write("-GPS:all\n")
                f.write("-XMP-drone-dji:all\n")
                f.write(str(dst) + "\n")
            elif mode == "all":
                f.write("-TagsFromFile\n")
                f.write(str(src) + "\n")
                f.write("-all:all\n")
                f.write("-icc_profile:all\n")
                f.write(str(dst) + "\n")
            else:
                raise ValueError(mode)

            f.write("-execute\n")

        f.write("-stay_open\nFalse\n")

    try:
        cmd = [exiftool, "-stay_open", "True", "-@", arg_path]
        if dry_run:
            print("[DRY RUN] exiftool stay_open cmd:", " ".join(cmd))
            return

        proc = subprocess.run(cmd, capture_output=True, check=False)
        stderr = proc.stderr.decode("utf-8", errors="replace").strip()
        if proc.returncode != 0:
            raise RuntimeError(f"exiftool failed rc={proc.returncode}\n{stderr}")
        if stderr:
            # -q -q 仍可能输出警告
            print("[EXIFTOOL STDERR]")
            print(stderr)

    finally:
        try:
            os.unlink(arg_path)
        except OSError:
            pass


def cleanup_originals_for_pairs(pairs: List[Tuple[Path, Path]], dry_run: bool) -> Tuple[int, int]:
    """
    删除 dst 同目录下的 <dst.name>_original 备份文件。
    返回：(found, deleted)
    """
    found = 0
    deleted = 0
    seen = set()
    for _, dst in pairs:
        # 防止重复
        key = str(dst)
        if key in seen:
            continue
        seen.add(key)

        bkp = dst.parent / (dst.name + "_original")
        if bkp.exists() and bkp.is_file():
            found += 1
            if dry_run:
                continue
            try:
                bkp.unlink()
                deleted += 1
            except Exception:
                # 不直接失败，给出提示
                print(f"[WARN] failed to delete backup: {bkp}")
    return found, deleted


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def exiftool_gps_dump(
    exiftool: str,
    files: List[Path],
) -> Dict[str, Tuple[Optional[float], Optional[float], Optional[float]]]:
    if not files:
        return {}

    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as f:
        list_path = f.name
        for p in files:
            f.write(str(p) + "\n")

    cmd = [
        exiftool, "-q", "-q",
        "-json", "-n",
        "-GPSLatitude", "-GPSLongitude", "-GPSAltitude",
        "-XMP-drone-dji:GPSLatitude", "-XMP-drone-dji:GPSLongitude", "-XMP-drone-dji:AbsoluteAltitude",
        "-@", list_path,
    ]
    proc = subprocess.run(cmd, capture_output=True, check=False)
    os.unlink(list_path)

    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.decode("utf-8", errors="replace"))

    text = proc.stdout.decode("utf-8", errors="replace")
    try:
        recs = json.loads(text)
    except json.JSONDecodeError:
        s = text.find("[")
        e = text.rfind("]")
        if s >= 0 and e > s:
            recs = json.loads(text[s:e + 1])
        else:
            raise

    out: Dict[str, Tuple[Optional[float], Optional[float], Optional[float]]] = {}
    for r in recs:
        srcfile = r.get("SourceFile", "")
        key = os.path.normpath(os.path.abspath(srcfile))
        if os.name == "nt":
            key = os.path.normcase(key)

        lat = r.get("GPSLatitude", None)
        lon = r.get("GPSLongitude", None)
        alt = r.get("GPSAltitude", None)

        if alt is None:
            aa = r.get("AbsoluteAltitude", None)
            if aa is not None:
                try:
                    alt = float(aa)
                except Exception:
                    alt = None

        def _to_f(x):
            if x is None:
                return None
            try:
                return float(x)
            except Exception:
                return None

        out[key] = (_to_f(lat), _to_f(lon), _to_f(alt))
    return out


def verify_gps(
    exiftool: str,
    pairs: List[Tuple[Path, Path]],
    tol_xy_m: float,
    tol_alt_m: float,
    report_csv: Optional[Path],
    report_all: bool,
) -> int:
    src_files = [s for s, _ in pairs]
    dst_files = [d for _, d in pairs]

    src_map = exiftool_gps_dump(exiftool, src_files)
    dst_map = exiftool_gps_dump(exiftool, dst_files)

    mismatches: List[Tuple] = []
    all_rows: List[Tuple] = []

    for s, d in pairs:
        sk = _norm_path(s)
        dk = _norm_path(d)

        slat, slon, salt = src_map.get(sk, (None, None, None))
        dlat, dlon, dalt = dst_map.get(dk, (None, None, None))

        dist = None
        alt_diff = None
        reason = "ok"

        if slat is None or slon is None or dlat is None or dlon is None:
            reason = "missing_latlon"
        else:
            dist = haversine_m(slat, slon, dlat, dlon)
            if salt is not None and dalt is not None:
                alt_diff = abs(salt - dalt)
            if dist > tol_xy_m or (alt_diff is not None and alt_diff > tol_alt_m):
                reason = "mismatch"

        row = (d.name, slat, slon, salt, dlat, dlon, dalt, dist, alt_diff, reason)
        all_rows.append(row)
        if reason != "ok":
            mismatches.append(row)

    print(f"[VERIFY] pairs={len(pairs)} mismatches={len(mismatches)} tol_xy_m={tol_xy_m} tol_alt_m={tol_alt_m}")

    if mismatches:
        print("[VERIFY] Examples:")
        for row in mismatches[:20]:
            fn, slat, slon, salt, dlat, dlon, dalt, dist, alt_diff, reason = row
            print(f"  - {fn}: src=({slat},{slon},{salt}) dst=({dlat},{dlon},{dalt}) "
                  f"dist_m={None if dist is None else round(dist,3)} alt_diff_m={alt_diff} reason={reason}")

    if report_csv is not None:
        report_csv.parent.mkdir(parents=True, exist_ok=True)
        with report_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["filename", "src_lat", "src_lon", "src_alt",
                        "dst_lat", "dst_lon", "dst_alt", "dist_m", "alt_diff_m", "reason"])
            rows_to_write = all_rows if report_all else mismatches
            for row in rows_to_write:
                w.writerow(row)
        print(f"[VERIFY] wrote report: {report_csv} (rows={'ALL' if report_all else 'mismatch_only'}={len(all_rows) if report_all else len(mismatches)})")

    return len(mismatches)


def main() -> int:
    ap = argparse.ArgumentParser(description="Transfer EXIF/XMP metadata from src_dir to dst_dir and verify GPS.")
    ap.add_argument("--src_dir", required=True)
    ap.add_argument("--dst_dir", required=True)
    ap.add_argument("--recursive", action="store_true")
    ap.add_argument("--ext", default=",".join(IMG_EXTS_DEFAULT))
    ap.add_argument("--match", choices=["basename", "relative"], default="basename")
    ap.add_argument("--case_insensitive", action="store_true")
    ap.add_argument("--exiftool", default="exiftool")

    ap.add_argument("--keep_backup", action="store_true", help="Keep *_original backups.")
    ap.add_argument("--cleanup_originals", dest="cleanup_originals", action="store_true", help="Delete <dst>_original after copy (default).")
    ap.add_argument("--no_cleanup_originals", dest="cleanup_originals", action="store_false", help="Do not delete backups.")
    ap.set_defaults(cleanup_originals=True)

    ap.add_argument("--chunk", type=int, default=300)
    ap.add_argument("--dry_run", action="store_true")

    ap.add_argument("--mode", choices=["all", "gps"], default="all")
    ap.add_argument("--clear_dst_gps", action="store_true")

    ap.add_argument("--verify_gps", action="store_true")
    ap.add_argument("--verify_tol_xy_m", type=float, default=2.0)
    ap.add_argument("--verify_tol_alt_m", type=float, default=5.0)
    ap.add_argument("--verify_report", default="")
    ap.add_argument("--verify_report_all", action="store_true")

    ap.add_argument("--print_mapping", type=int, default=10)

    args = ap.parse_args()

    src_dir = Path(args.src_dir).expanduser().resolve()
    dst_dir = Path(args.dst_dir).expanduser().resolve()
    if not src_dir.exists():
        print(f"[ERROR] src_dir not found: {src_dir}", file=sys.stderr)
        return 2
    if not dst_dir.exists():
        print(f"[ERROR] dst_dir not found: {dst_dir}", file=sys.stderr)
        return 2

    exts = [e.strip() for e in args.ext.split(",") if e.strip()]
    case_ins = args.case_insensitive or (os.name == "nt")

    src_map, src_dups = build_index(src_dir, args.recursive, exts, args.match, case_ins)
    dst_map, dst_dups = build_index(dst_dir, args.recursive, exts, args.match, case_ins)

    if src_dups:
        print(f"[WARN] src duplicate keys={len(src_dups)} under match={args.match}")
    if dst_dups:
        print(f"[WARN] dst duplicate keys={len(dst_dups)} under match={args.match}")

    pairs: List[Tuple[Path, Path]] = []
    missing = 0

    for k, d in dst_map.items():
        if k in src_dups:
            continue
        s = src_map.get(k)
        if s is None:
            missing += 1
            continue
        pairs.append((s, d))

    for k, ds in dst_dups.items():
        if k in src_dups:
            continue
        s = src_map.get(k)
        if s is None:
            missing += len(ds)
            continue
        for d in ds:
            pairs.append((s, d))

    print(f"[INFO] matched_pairs={len(pairs)} missing_src={missing}")

    if args.print_mapping > 0 and pairs:
        print("[INFO] Mapping examples (src -> dst):")
        for s, d in pairs[:args.print_mapping]:
            print(f"  - {s} -> {d}")

    # copy
    chunks = chunk_list(pairs, int(args.chunk))
    for i, ch in enumerate(chunks, 1):
        print(f"[INFO] exiftool stay_open batch {i}/{len(chunks)}: {len(ch)} pairs")
        run_exiftool_stay_open(
            exiftool=args.exiftool,
            pairs=ch,
            mode=args.mode,
            clear_dst_gps=args.clear_dst_gps,
            keep_backup=args.keep_backup,
            dry_run=args.dry_run,
        )

    print("[OK] exiftool copy finished.")

    # cleanup backups
    if (not args.keep_backup) and args.cleanup_originals:
        found, deleted = cleanup_originals_for_pairs(pairs, dry_run=args.dry_run)
        if args.dry_run:
            print(f"[CLEANUP] would remove backups: found={found}")
        else:
            print(f"[CLEANUP] removed backups: found={found} deleted={deleted}")

    # verify
    if args.verify_gps and (not args.dry_run):
        report = Path(args.verify_report).expanduser().resolve() if args.verify_report else None
        mism = verify_gps(
            exiftool=args.exiftool,
            pairs=pairs,
            tol_xy_m=float(args.verify_tol_xy_m),
            tol_alt_m=float(args.verify_tol_alt_m),
            report_csv=report,
            report_all=bool(args.verify_report_all),
        )
        if mism > 0:
            return 3

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
