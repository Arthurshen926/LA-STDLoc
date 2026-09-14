#!/usr/bin/env python3
"""Generate a read-only, file-level LaFGS/STDLoc archive plan.

The plan is intentionally non-destructive.  It inventories the remaining
top-level ``lafgs``/``stdloc`` entries under ``/mnt/pool/sqy`` and a small set
of explicitly protected current trees, assigns KEEP/ARCHIVE/DELETE_CANDIDATE,
and computes a deterministic destination below ``/mnt/pool/sqy/LaFGS_archive``
for every ARCHIVE row.
"""

from __future__ import annotations

import os
import re
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


SQY = Path("/mnt/pool/sqy")
ARCHIVE_ROOT = SQY / "LaFGS_archive"
OUT_MD = Path("/root/STDLoc/docs/lafgs_archive_full_dry_run_20260909.md")
OUT_TSV = Path("/root/STDLoc/docs/lafgs_archive_full_dry_run_20260909.tsv")


TEXT_EXTS = {
    ".json",
    ".jsonl",
    ".csv",
    ".tsv",
    ".yaml",
    ".yml",
    ".md",
    ".txt",
    ".log",
    ".sh",
    ".cfg",
    ".ini",
    ".toml",
    ".xml",
    ".html",
    ".command",
}
REBUILDABLE_EXTS = {
    ".pt",
    ".pth",
    ".pth.tar",
    ".ckpt",
    ".npz",
    ".npy",
    ".pkl",
    ".pickle",
    ".path",
    ".bin",
    ".cache",
    ".ply",
}
FIGURE_EXTS = {".png", ".jpg", ".jpeg", ".pdf", ".svg"}
DATE_RE = re.compile(r"(?<!\d)(20\d{6})(?!\d)")


@dataclass(frozen=True)
class RootSpec:
    name: str
    path: Path
    kind: str
    description: str
    default_action: str | None = None
    archive_category: str = "historical"


def parse_date(name: str) -> str:
    match = DATE_RE.search(name)
    if match:
        raw = match.group(1)
        return f"{raw[:4]}-{raw[4:6]}-{raw[6:]}"
    return "undated"


def is_path_file(path: Path) -> bool:
    try:
        return path.is_file() or path.is_symlink()
    except OSError:
        return True


def root_specs() -> list[RootSpec]:
    specs: list[RootSpec] = []
    names: set[str] = set()
    for path in sorted(SQY.iterdir(), key=lambda p: p.name.lower()):
        low = path.name.lower()
        if path.name == ARCHIVE_ROOT.name:
            # A rerun must not recursively scan the destination it is planning.
            continue
        if "lafgs" not in low and "stdloc" not in low:
            continue
        names.add(path.name)
        if path.name == "Cambridge_stdloc":
            specs.append(
                RootSpec(
                    path.name,
                    path,
                    "protected_data",
                    "当前 Cambridge 原始/处理后数据；AnyGSLoc 与 v23 配置仍直接引用",
                    "KEEP",
                    "protected",
                )
            )
        elif path.name == "lafgs_v24_cambridge_high_capacity_online_20260901":
            specs.append(
                RootSpec(
                    path.name,
                    path,
                    "protected_current",
                    "当前 AnyGSLoc/v24-v26 artifact_root；保留所有现有 map/cache",
                    "KEEP",
                    "protected",
                )
            )
        elif path.name == "stdloc_lafgs_v1_frozen_multiscene_20260731":
            specs.append(
                RootSpec(
                    path.name,
                    path,
                    "protected_prior",
                    "五个 Cambridge 的 MAtCha 2DGS 主线先验及其 manifest",
                    "KEEP",
                    "protected",
                )
            )
        elif path.name == "stdloc_lafgs_offtheshelf_prior_20260802":
            specs.append(
                RootSpec(
                    path.name,
                    path,
                    "prior_ablation",
                    "Cambridge vanilla 2DGS/3DGS 消融先验；训练副产物按文件拆分",
                    None,
                    "prior_ablation",
                )
            )
        elif path.name == "stdloc_closed_loop_archive_20260909":
            specs.append(
                RootSpec(
                    path.name,
                    path,
                    "legacy_archive_migration",
                    "上一轮闭环清理产生的旧归档；迁移到统一 LaFGS_archive",
                    None,
                    "legacy_archive_migration",
                )
            )
        elif path.name in {
            "lafgs_assets",
            "lafgs_experiment_tools",
            "third_party_lafgs_priors",
        }:
            specs.append(
                RootSpec(
                    path.name,
                    path,
                    "tooling_dependency",
                    "仍可能用于重建先验或实验的工具/第三方代码",
                    "KEEP",
                    "tooling",
                )
            )
        elif path.name == "lafgs_model_cache":
            specs.append(
                RootSpec(
                    path.name,
                    path,
                    "duplicate_tool_cache",
                    "与 lafgs_assets 中的 SuperPoint 权重重复；默认归档为可恢复副本",
                    None,
                    "tooling",
                )
            )
        elif path.name in {"lafgs_indoor_sfm_env", "lafgs_indoor_sfm_py39"}:
            specs.append(
                RootSpec(
                    path.name,
                    path,
                    "historical_environment",
                    "旧 indoor SFM 环境；当前没有运行进程，完整保留到统一归档",
                    None,
                    "tooling_history",
                )
            )
        elif path.name == "lafgs_paper_figures_20260806":
            specs.append(
                RootSpec(
                    path.name,
                    path,
                    "historical_paper_figures",
                    "旧论文图表及渲染缓存；保留到统一归档以便查阅",
                    None,
                    "paper_figures",
                )
            )
        elif path.name in {
            "lafgs_projective_anchor_v4_cambridge_full_20260817",
            "lafgs_render_track_only_fullchain_20260814",
        }:
            specs.append(
                RootSpec(
                    path.name,
                    path,
                    "closed_loop_reference",
                    "v23 闭环配置仍引用的历史 map/metric；迁移后路径需更新才可重跑",
                    None,
                    "closed_loop_maps",
                )
            )
        else:
            specs.append(
                RootSpec(
                    path.name,
                    path,
                    "historical_output",
                    "旧 STDLoc/LaFGS 实验输出；保留小型记录，二进制中间产物列为删除候选",
                    None,
                    "stdloc_history" if low.startswith("stdloc") else "lafgs_history",
                )
            )

    # Explicitly include current AnyGSLoc trees even though their names do not
    # all contain lafgs/stdloc.  G4Splat_runs is deliberately not scanned.
    for name, description in {
        "anygsloc_paper_experiments_20260902": "当前 AnyGSLoc 论文主线/消融输出；用户明确要求保持原位",
        "anygsloc_execution_queue_20260909": "当前 AnyGSLoc 执行队列与正在生成的 map/evaluation；保持原位",
    }.items():
        path = SQY / name
        if path.exists() and name not in names:
            specs.append(RootSpec(name, path, "protected_current", description, "KEEP", "protected"))

    # The previous AnySplat cleanup also needs to cover this nested model
    # cache, which is not a top-level lafgs/stdloc directory.
    nested_anysplat = SQY / "huggingface_cache/hub/models--lhjiang--anysplat"
    if nested_anysplat.exists():
        specs.append(
            RootSpec(
                str(nested_anysplat.relative_to(SQY)),
                nested_anysplat,
                "obsolete_anysplat",
                "用户已明确要求清理的旧 AnySplat HuggingFace cache",
                None,
                "obsolete_anysplat",
            )
        )
    return specs


def disk_bytes(path: Path) -> int:
    try:
        output = subprocess.check_output(["du", "-sB1", "--", str(path)], text=True)
        return int(output.split()[0])
    except (OSError, ValueError, subprocess.CalledProcessError):
        return 0


def lstat_info(path: Path) -> tuple[str, int, int]:
    try:
        st = path.lstat()
    except OSError:
        return "unreadable", 0, 0
    if path.is_symlink():
        return "symlink", st.st_size, st.st_blocks * 512
    if path.is_dir():
        return "dir", 0, st.st_blocks * 512
    return "file", st.st_size, st.st_blocks * 512


def iter_entries(spec: RootSpec):
    path = spec.path
    kind, apparent, allocated = lstat_info(path)
    if kind in {"file", "symlink", "unreadable"}:
        yield path, ".", kind, apparent, allocated
        return
    if kind != "dir":
        return
    for dirpath, dirnames, filenames in os.walk(path, followlinks=False):
        # Keep symlinked directories as individual rows and do not traverse.
        retained_dirs: list[str] = []
        for dirname in dirnames:
            child = Path(dirpath) / dirname
            child_kind, child_apparent, child_allocated = lstat_info(child)
            if child_kind == "symlink":
                yield child, child.relative_to(path).as_posix(), child_kind, child_apparent, child_allocated
            else:
                retained_dirs.append(dirname)
        dirnames[:] = retained_dirs
        for filename in filenames:
            child = Path(dirpath) / filename
            child_kind, child_apparent, child_allocated = lstat_info(child)
            yield child, child.relative_to(path).as_posix(), child_kind, child_apparent, child_allocated


def generic_historical_action(
    relative: str,
    kind: str,
    apparent: int,
    allocated: int,
    source_path: Path | None = None,
) -> tuple[str, str]:
    low = relative.lower()
    name = Path(relative).name.lower()
    suffix = Path(relative).suffix.lower()
    if kind == "symlink":
        if source_path is not None and source_path.exists():
            return "ARCHIVE", "历史符号链接；归档链接记录"
        return "DELETE_CANDIDATE", "失效符号链接，目标已不存在"

    # Preserve final/evidence artifacts in the archive, while treating
    # checkpoints, feature/query/render caches and training tensors as
    # rebuildable deletion candidates.
    rebuild_tokens = (
        "checkpoint",
        "query_cache",
        "feature_cache",
        "rendered_rgb",
        "observation",
        "track_payload",
        "function_graph",
        "visibility",
        "raster",
        "teacher",
        "descriptor",
        "embedding",
        "cache",
        "shard",
        "events.out",
        "tensorboard",
        "loc_state",
    )
    evidence_tokens = (
        "report",
        "summary",
        "manifest",
        "registry",
        "contract",
        "audit",
        "result",
        "provenance",
        "metric",
        "decision",
        "analysis",
        "protocol",
        "aggregate",
        "config",
        "receipt",
        "README".lower(),
    )
    figure_tokens = ("paper", "figure", "panel", "plot", "visual", "diagnostic")

    if any(token in low for token in rebuild_tokens) and not any(token in low for token in evidence_tokens):
        return "DELETE_CANDIDATE", "可由历史配置/原始数据重建的 checkpoint、cache 或中间张量"
    if suffix in REBUILDABLE_EXTS:
        if ("point_cloud/iteration_30000" in low or "final" in low or "prior" in low or "projective_map" in low or "anchor_map" in low) and not any(token in low for token in rebuild_tokens):
            return "ARCHIVE", "历史最终先验/map 二进制；迁移到统一归档以保留实验记录"
        if suffix == ".ply" and ("prior" in low or "native" in low):
            return "ARCHIVE", "历史先验 PLY；保留归档副本"
        return "DELETE_CANDIDATE", "可重建的历史二进制中间产物或 checkpoint"
    if suffix in FIGURE_EXTS and any(token in low for token in figure_tokens):
        return "ARCHIVE", "论文/诊断图像；保留到统一归档"
    if suffix in TEXT_EXTS:
        if apparent > 200 * 1024**2 or allocated > 200 * 1024**2:
            return "DELETE_CANDIDATE", "超大 raw diagnostic/text dump，可由历史 run 重建"
        return "ARCHIVE", "小型实验记录、配置、结果或 provenance"
    if suffix in FIGURE_EXTS:
        return "DELETE_CANDIDATE", "非论文核心的历史渲染/诊断图像，可重建"
    if apparent <= 50 * 1024**2:
        return "ARCHIVE", "小型历史辅助文件；集中归档"
    return "DELETE_CANDIDATE", "大型历史中间文件；当前主线不依赖"


def classify(
    spec: RootSpec,
    relative: str,
    kind: str,
    apparent: int,
    allocated: int,
    source_path: Path | None = None,
) -> tuple[str, str]:
    low = relative.lower()

    # AnySplat outputs/caches were explicitly retired.  Keep only the small
    # cleanup receipt itself as historical metadata; all other AnySplat paths
    # are deletion candidates even when embedded in a paper/release root.
    if spec.kind == "obsolete_anysplat":
        return "DELETE_CANDIDATE", "旧 AnySplat HuggingFace cache；用户已明确要求清除"
    if "anysplat" in low and "receipt" not in low and "cleanup" not in low:
        return "DELETE_CANDIDATE", "旧 AnySplat 内容；用户已明确要求清除"

    if spec.default_action == "KEEP":
        return "KEEP", spec.description

    if spec.kind == "duplicate_tool_cache":
        if Path(relative).name == "superpoint_v1.pth":
            return "ARCHIVE", "与 lafgs_assets/superpoint_v1.pth 内容相同的重复权重"
        return "ARCHIVE", "重复/旧模型 cache，集中归档"

    if spec.kind == "prior_ablation":
        # These four final vanilla prior trees are the explicitly retained
        # Cambridge ablation priors.  Everything else is rebuildable support.
        keep_prefixes = tuple(
            f"priors/{scene}/{method}/"
            for scene in ("KingsCollege", "OldHospital")
            for method in ("vanilla_2dgs", "vanilla_3dgs")
        )
        if low.startswith(tuple(prefix.lower() for prefix in keep_prefixes)):
            if "/point_cloud/iteration_30000/point_cloud.ply" in low or low.endswith("rgb_prior_manifest.json") or low.endswith("prior_quality.json") or low.endswith("offtheshelf_prior_protocol.json"):
                return "KEEP", "当前 Cambridge vanilla 2DGS/3DGS 消融先验或其质量/协议 manifest"
            if "/native/point_cloud/iteration_30000/point_cloud.ply" in low:
                return "KEEP", "当前 Cambridge vanilla 原生最终 PLY"
            if "/stdloc_model/point_cloud/iteration_30000/point_cloud.ply" in low:
                return "KEEP", "当前 AnyGSLoc 消融使用的 stdloc_model 最终 PLY"
            if Path(relative).suffix.lower() in TEXT_EXTS and any(t in low for t in ("quality", "protocol", "manifest", "cfg_args", "command")):
                return "ARCHIVE", "消融先验 provenance/命令记录"
            if Path(relative).suffix.lower() in {".log", ".tfevents"} or "events.out" in low:
                return "DELETE_CANDIDATE", "训练日志/事件文件，可重建"
            return "DELETE_CANDIDATE", "vanilla prior 训练副产物，可从保留 PLY 重新生成"
        return generic_historical_action(relative, kind, apparent, allocated, source_path)

    if spec.kind == "closed_loop_reference":
        if kind == "symlink":
            return "ARCHIVE", "历史归档中的符号链接记录"
        # v23 still names these exact small map/metric/report files.  Preserve
        # them in the unified archive, while allowing feature/appearance/cache
        # tensors in the same root to become deletion candidates.
        basename = Path(relative).name.lower()
        if basename in {
            "rendered_rgb_track_map.pt",
            "rendered_rgb_track_map.json",
            "metric_state_identity.pt",
            "identity_metric.pt",
            "identity_metric.json",
        }:
            return "ARCHIVE", "v23 配置引用的历史 map/metric/report；迁移后需更新绝对路径"
        return generic_historical_action(relative, kind, apparent, allocated, source_path)

    if spec.kind in {"historical_paper_figures", "legacy_archive_migration"}:
        if kind == "symlink":
            return "ARCHIVE", "历史归档中的符号链接记录"
        return "ARCHIVE", spec.description

    if spec.kind == "historical_environment":
        return "ARCHIVE", "旧 indoor SFM 环境；集中归档，不再作为当前运行环境"

    return generic_historical_action(relative, kind, apparent, allocated, source_path)


def archive_destination(spec: RootSpec, relative: str) -> str:
    date = parse_date(spec.name)
    if relative == "." and not spec.path.is_dir():
        tail = spec.name
    else:
        tail = "root" if relative == "." else relative
    return str(ARCHIVE_ROOT / spec.archive_category / date / spec.name / tail)


def collect_references(names: list[str]) -> dict[str, dict[str, set[str]]]:
    """Collect source/config/doc references without treating generated reports as refs."""
    result = {name: {"config": set(), "script": set(), "doc": set()} for name in names}
    try:
        proc = subprocess.run(
            [
                "rg",
                "-n",
                "--hidden",
                "-g",
                "!*.pyc",
                "-g",
                "!docs/stdloc_cleanup_dry_run*",
                "-g",
                "!docs/stdloc_cleanup_execution*",
                "-g",
                "!docs/lafgs_archive_full_dry_run*",
                "/mnt/pool/sqy/",
                "/root/STDLoc",
            ],
            capture_output=True,
            text=True,
            errors="replace",
            check=False,
        )
    except OSError:
        return result
    tokens = [(name, f"/mnt/pool/sqy/{name}") for name in names]
    for line in proc.stdout.splitlines():
        if ":" not in line:
            continue
        source, content = line.split(":", 1)
        for name, token in tokens:
            if token not in content:
                continue
            if "/configs/" in source:
                bucket = "config"
            elif "/scripts/" in source and "cleanup" not in source:
                bucket = "script"
            else:
                bucket = "doc"
            result[name][bucket].add(source)
    return result


def fmt_bytes(value: int) -> str:
    return f"{value / 1024**3:.2f} GiB"


def main() -> None:
    generated = datetime.now().astimezone().isoformat(timespec="seconds")
    specs = root_specs()
    refs = collect_references([spec.name for spec in specs])
    rows: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []

    for spec in specs:
        counts: Counter[str] = Counter()
        bytes_by_action: Counter[str] = Counter()
        file_count = 0
        apparent_total = 0
        allocated_total = disk_bytes(spec.path)
        for path, relative, kind, apparent, allocated in iter_entries(spec):
            action, reason = classify(spec, relative, kind, apparent, allocated, path)
            destination = archive_destination(spec, relative) if action == "ARCHIVE" else ""
            row = {
                "source_root": str(spec.path),
                "root_name": spec.name,
                "scope_type": kind,
                "relative_path": relative,
                "source_path": str(path),
                "apparent_bytes": apparent,
                "allocated_bytes": allocated,
                "action": action,
                "archive_destination": destination,
                "reason": reason,
            }
            rows.append(row)
            file_count += 1
            apparent_total += apparent
            counts[action] += 1
            bytes_by_action[action] += allocated
        if file_count == 0 and spec.path.is_dir():
            # Make empty historical roots visible in the directory-level plan;
            # otherwise they would silently survive a file-only execution.
            action = "KEEP" if spec.default_action == "KEEP" else "DELETE_CANDIDATE"
            reason = spec.description if action == "KEEP" else "空的历史目录；文件清理后可安全移除"
            allocated = disk_bytes(spec.path)
            rows.append(
                {
                    "source_root": str(spec.path),
                    "root_name": spec.name,
                    "scope_type": "tree",
                    "relative_path": ".",
                    "source_path": str(spec.path),
                    "apparent_bytes": 0,
                    "allocated_bytes": allocated,
                    "action": action,
                    "archive_destination": "",
                    "reason": reason,
                }
            )
            file_count = 1
            counts[action] += 1
            bytes_by_action[action] += allocated
        summaries.append(
            {
                "spec": spec,
                "files": file_count,
                "apparent": apparent_total,
                "allocated": allocated_total,
                "counts": counts,
                "bytes": bytes_by_action,
                "refs": refs.get(spec.name, {"config": set(), "script": set(), "doc": set()}),
            }
        )

    OUT_TSV.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "source_root",
        "root_name",
        "scope_type",
        "relative_path",
        "source_path",
        "apparent_bytes",
        "allocated_bytes",
        "action",
        "archive_destination",
        "reason",
    ]
    with OUT_TSV.open("w", encoding="utf-8") as handle:
        handle.write("\t".join(columns) + "\n")
        for row in sorted(rows, key=lambda item: (str(item["root_name"]).lower(), str(item["relative_path"]).lower())):
            handle.write(
                "\t".join(str(row[column]).replace("\t", " ").replace("\n", " ") for column in columns) + "\n"
            )

    total_counts: Counter[str] = Counter()
    total_bytes: Counter[str] = Counter()
    for summary in summaries:
        total_counts.update(summary["counts"])
        total_bytes.update(summary["bytes"])

    lines: list[str] = []
    lines.append("# LaFGS / STDLoc full archive dry-run")
    lines.append("")
    lines.append(f"Generated: `{generated}`")
    lines.append("")
    lines.append("本报告只读扫描并生成逐文件清单；没有创建、移动、删除或修改 `/mnt/pool/sqy` 下的任何实验文件。")
    lines.append("")
    lines.append("统一归档目标：`/mnt/pool/sqy/LaFGS_archive/<功能>/<日期>/<原目录>/<相对路径>`。实际执行前仍需再次确认；`G4Splat_runs` 与 `/mnt/pool/sqy/datasets` 原始数据树不在移动/删除范围内。")
    lines.append("")
    lines.append("动作含义：`KEEP` 原位保留；`ARCHIVE` 移入统一 LaFGS_archive；`DELETE_CANDIDATE` 可重建的删除候选，当前不执行。")
    lines.append("")
    lines.append("## Overall summary")
    lines.append("")
    lines.append(f"- 扫描 root：**{len(specs)}**；逐条文件/符号链接记录：**{len(rows)}**。")
    lines.append(f"- KEEP：{total_counts['KEEP']} 条，{fmt_bytes(total_bytes['KEEP'])} allocated。")
    lines.append(f"- ARCHIVE：{total_counts['ARCHIVE']} 条，{fmt_bytes(total_bytes['ARCHIVE'])} allocated。")
    lines.append(f"- DELETE_CANDIDATE：{total_counts['DELETE_CANDIDATE']} 条，{fmt_bytes(total_bytes['DELETE_CANDIDATE'])} allocated。")
    lines.append("")
    lines.append("## Root summary")
    lines.append("")
    lines.append("| root | allocated | records | KEEP | ARCHIVE | DELETE_CANDIDATE | refs |")
    lines.append("|---|---:|---:|---:|---:|---:|---|")
    for summary in sorted(summaries, key=lambda item: str(item["spec"].name).lower()):
        spec: RootSpec = summary["spec"]
        counts: Counter[str] = summary["counts"]
        bytes_map: Counter[str] = summary["bytes"]
        ref_map: dict[str, set[str]] = summary["refs"]
        refs_text = ", ".join(
            f"{bucket}:{len(ref_map[bucket])}"
            for bucket in ("config", "script", "doc")
            if ref_map[bucket]
        ) or "none"
        lines.append(
            f"| `{spec.name}` | {fmt_bytes(int(summary['allocated']))} | {summary['files']} | "
            f"{counts['KEEP']} / {fmt_bytes(bytes_map['KEEP'])} | "
            f"{counts['ARCHIVE']} / {fmt_bytes(bytes_map['ARCHIVE'])} | "
            f"{counts['DELETE_CANDIDATE']} / {fmt_bytes(bytes_map['DELETE_CANDIDATE'])} | {refs_text} |"
        )
    lines.append("")
    lines.append("## Protection and migration notes")
    lines.append("")
    lines.append("- `anygsloc_paper_experiments_20260902`、`anygsloc_execution_queue_20260909`、`Cambridge_stdloc`、五个 MAtCha 先验、`lafgs_v24_cambridge_high_capacity_online_20260901` 均按 KEEP 处理。")
    lines.append("- `stdloc_lafgs_offtheshelf_prior_20260802` 只保留四个 Cambridge vanilla 2DGS/3DGS 最终先验 PLY、manifest、quality/protocol；训练日志、事件文件、重复数据集副本会分别归档或列为删除候选。")
    lines.append("- v23 闭环 map/metric 所在目录按 ARCHIVE 迁移；迁移后若要重新运行 v23，需要同步更新配置中的绝对路径。")
    lines.append("- 旧 `stdloc_closed_loop_archive_20260909` 计划迁入 `LaFGS_archive/legacy_archive_migration/2026-09-09/`，避免继续保留多个归档入口。")
    lines.append("- 5 个顶层 `lafgs_*_oh` 失效符号链接会列为 DELETE_CANDIDATE，不会被复制成新的归档垃圾。")
    lines.append("- 旧 AnySplat 内容（包括 `lafgs_paper_release_gate_20260803` 下的残留 prior、嵌套 HuggingFace cache）列为 DELETE_CANDIDATE；只保留上一轮清理 receipt 作为归档记录。")
    lines.append("- `docs/` 中对旧实验路径的引用只作为历史证据；不等于这些二进制仍是运行时依赖。")
    lines.append("")
    lines.append("## File-level manifest")
    lines.append("")
    lines.append(f"完整 TSV：[`lafgs_archive_full_dry_run_20260909.tsv`](/root/STDLoc/docs/lafgs_archive_full_dry_run_20260909.tsv)。字段包含源路径、动作、原因和统一归档目标路径。")
    lines.append("")
    lines.append("## Not scanned")
    lines.append("")
    lines.append("- `/mnt/pool/sqy/G4Splat_runs`：按用户要求保持不动。")
    lines.append("- `/mnt/pool/sqy/datasets`：作为原始/处理后数据树保持不动；仅在清单中通过当前 AnyGSLoc 路径说明其受保护状态。")
    lines.append("")

    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(OUT_MD)
    print(OUT_TSV)
    print(f"rows={len(rows)} KEEP={total_counts['KEEP']} ARCHIVE={total_counts['ARCHIVE']} DELETE_CANDIDATE={total_counts['DELETE_CANDIDATE']}")


if __name__ == "__main__":
    main()
