#!/usr/bin/env python3
"""Generate a read-only, file-level STDLoc/LaFGS cleanup dry-run.

This script only reads the selected output roots and writes a Markdown summary
plus a TSV manifest.  It never deletes, moves, or modifies files under
/mnt/pool/sqy.
"""

from __future__ import annotations

import os
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


OUT_MD = Path("/root/STDLoc/docs/stdloc_cleanup_dry_run_20260909_v2.md")
OUT_TSV = Path("/root/STDLoc/docs/stdloc_cleanup_dry_run_20260909_v2.tsv")


@dataclass(frozen=True)
class RootSpec:
    key: str
    path: Path
    description: str


ROOTS = [
    RootSpec(
        "anygsloc_paper",
        Path("/mnt/pool/sqy/anygsloc_paper_experiments_20260902"),
        "当前 AnyGSLoc 论文主线/消融输出；不整目录删除",
    ),
    RootSpec(
        "legacy_archive",
        Path("/mnt/pool/sqy/stdloc_lafgs_legacy_archive_20260801"),
        "旧 LaFGS final-state 模型库",
    ),
    RootSpec(
        "rgb_sanitization",
        Path("/mnt/pool/sqy/stdloc_lafgs_rgb_prior_sanitization_20260725"),
        "OldHospital RGB prior 清洗/sky-no-sky 历史实验",
    ),
    RootSpec(
        "v6_mapping_accuracy",
        Path("/mnt/pool/sqy/lafgs_v6_full_mapping_accuracy_20260823"),
        "v6 mapping accuracy 与 invalid closed-loop 诊断",
    ),
    RootSpec(
        "prior_robust_a1",
        Path("/mnt/pool/sqy/stdloc_lafgs_prior_robust_a1_20260801"),
        "OldHospital sky/no-sky prior robustness 历史实验",
    ),
    RootSpec(
        "clean_rebuild",
        Path("/mnt/pool/sqy/stdloc_lafgs_clean_rebuild_20260728_v1"),
        "OldHospital clean rebuild 与 selector/replay 历史实验",
    ),
    RootSpec(
        "phase2_archive",
        Path("/mnt/pool/sqy/stdloc_lafgs_phase2_archive_20260801"),
        "早期 LaFGS phase-2 模型变体归档",
    ),
    RootSpec(
        "rgb_prior_cache",
        Path("/mnt/pool/sqy/lafgs_rgb_prior_cache"),
        "独立 RGB prior query cache",
    ),
    RootSpec(
        "query_cache",
        Path("/mnt/pool/sqy/lafgs_query_cache"),
        "独立 12Scenes query cache",
    ),
    RootSpec(
        "v6_feedback_core",
        Path("/mnt/pool/sqy/lafgs_v6_feedback_core_20260824"),
        "v6 feedback core 历史实验",
    ),
    RootSpec(
        "v6_feedback_control_20260824",
        Path("/mnt/pool/sqy/lafgs_v6_feedback_control_20260824"),
        "v6 feedback control 历史实验（20260824）",
    ),
    RootSpec(
        "v6_feedback_control_20260825",
        Path("/mnt/pool/sqy/lafgs_v6_feedback_control_20260825"),
        "v6 feedback control 历史实验（20260825）",
    ),
]


ADDITIONAL_ROOTS = [
    Path("/mnt/pool/sqy/lafgs_projective_anchor_v4_cambridge_full_20260817"),
    Path("/mnt/pool/sqy/lafgs_7scenes_undistorted_factor_20260809"),
    Path("/mnt/pool/sqy/lafgs_v24_cambridge_high_capacity_online_20260901"),
    Path("/mnt/pool/sqy/lafgs_v5_phase1_resource_oracle_20260821"),
    Path("/mnt/pool/sqy/lafgs_v7_full_continuous_mainline_20260827"),
    Path("/mnt/pool/sqy/stdloc_lafgs_offtheshelf_prior_20260802"),
]


def disk_bytes(path: Path) -> int:
    """Return allocated bytes as reported by du, without following changes."""
    try:
        out = subprocess.check_output(["du", "-sB1", str(path)], text=True)
        return int(out.split()[0])
    except (OSError, ValueError, subprocess.CalledProcessError):
        return 0


def file_rows(root: Path):
    for dirpath, _, filenames in os.walk(root, followlinks=False):
        for name in filenames:
            path = Path(dirpath) / name
            try:
                st = path.stat()
            except OSError:
                continue
            yield path, st.st_size, st.st_blocks * 512


def rel_parts(root: Path, path: Path) -> tuple[str, ...]:
    try:
        return path.relative_to(root).parts
    except ValueError:
        return (path.name,)


def classify(spec: RootSpec, path: Path, apparent: int, allocated: int) -> tuple[str, str]:
    rel = path.relative_to(spec.path).as_posix()
    parts = rel.split("/")
    low = rel.lower()
    name = path.name.lower()
    ext = path.suffix.lower()

    # Current AnyGSLoc outputs: preserve result/audit state, flag only
    # regenerable feature/map artifacts.
    if spec.key == "anygsloc_paper":
        if name == "rendered_rgb_feature_cache.pt":
            return "DELETE_CANDIDATE", "可由当前 prior/query 配置重新渲染的 RGB feature cache"
        if ext in {".pt", ".pth"} and (
            "projective_map" in parts
            or "v2_audit" in parts
            or any(k in name for k in ("projective_anchor", "association_graph", "identity_metric", "metric_state"))
        ):
            return "DELETE_CANDIDATE", "可重建的 projective map/metric/association 中间产物"
        if ext in {".pt", ".pth", ".npz"}:
            return "ARCHIVE", "当前主线二进制中间产物；先归档，待论文结果冻结后再决定"
        if "/evaluation/" in f"/{low}" or "/audits/" in f"/{low}" or name.endswith("_state.json") or "input_audit" in name:
            return "KEEP", "主线评估结果、审计或可恢复状态"
        if ext in {".json", ".csv", ".md"}:
            return "KEEP", "小型结果/配置/审计文本"
        return "ARCHIVE", "AnyGSLoc 辅助日志或非主线输出"

    # Legacy/phase2 model zoos: no current mainline prior reference was found.
    if spec.key in {"legacy_archive", "phase2_archive"}:
        if parts and parts[0] == "final_state" and ext in {".ply", ".pt", ".pth"}:
            return "DELETE_CANDIDATE", "未选中的历史 final-state 模型变体；应先保留少数代表模型"
        if parts and parts[0] in {"metadata", "viewpoint"}:
            return "ARCHIVE", "小型 provenance/视点/审计资料，可集中归档"
        if ext in {".json", ".csv", ".tsv", ".yaml", ".yml", ".md", ".txt", ".log"}:
            return "ARCHIVE", "历史实验 provenance/summary；不影响当前 AnyGSLoc"
        return "ARCHIVE", "旧模型库辅助文件，待人工确认"

    # RGB sanitization: final prior manifests/PLYs are small and useful;
    # run-time caches and localization states are reproducible.
    if spec.key == "rgb_sanitization":
        if parts and parts[0] == "OldHospital" and len(parts) >= 2 and parts[1] != "runs":
            if ext == ".ply" or name.endswith("manifest.json") or "summary" in name or name in {"cameras.json", "input.ply"}:
                return "KEEP", "小型 RGB prior、manifest 或 sanitization summary"
            if name == "loc_state.pt":
                return "DELETE_CANDIDATE", "训练期 localization checkpoint，不是当前主线先验"
        if "runs" in parts:
            if ext in {".pt", ".pth", ".pkl", ".path"}:
                return "DELETE_CANDIDATE", "可重建的 query/map/teacher/cache 二进制"
            if "stdloc_results" in parts or "summary" in name:
                return "ARCHIVE", "历史评估结果，保留小型 provenance"
            return "ARCHIVE", "历史 run 日志/配置"
        if ext in {".json", ".csv", ".log", ".yaml", ".yml", ".sh", ".txt"}:
            return "ARCHIVE", "历史 sanitization 记录"
        return "ARCHIVE", "旧 prior 清洗辅助文件"

    # v6 mapping accuracy: feedback branch and maps are reproducible; result
    # JSON/registry is the useful archival record.
    if spec.key == "v6_mapping_accuracy":
        if ext in {".pt", ".pth"}:
            return "DELETE_CANDIDATE", "可重建的 observation/projective map/feedback 二进制"
        if ext == ".json":
            if "results" in name or "registry" in name or "summary" in name or "contract" in name or "report" in name:
                return "KEEP", "mapping accuracy 结果、registry 或 deployment contract"
            return "ARCHIVE", "closed-loop 小型 provenance"
        return "ARCHIVE", "v6 历史辅助文件"

    # Prior robustness: retain text evaluation/provenance; drop full-res and
    # graph/map states. Compact/canonical derived states are archive-only.
    if spec.key == "prior_robust_a1":
        if ext in {".pt", ".pth", ".pkl"}:
            if any(k in low for k in ("query_cache", "function_graph", "map_state", "visibility", "raster_provenance")):
                return "DELETE_CANDIDATE", "可重建的 query/function-graph/map/raster cache"
            return "ARCHIVE", "旧 robustness 分支的派生状态；不属于当前主线"
        if ext in {".json", ".yaml", ".yml"}:
            return "KEEP", "evaluation/config/contract/provenance 文本"
        if ext in {".log", ".sh"}:
            return "ARCHIVE", "历史运行日志或命令"
        return "ARCHIVE", "旧 prior robustness 辅助文件"

    # Clean rebuild: small summary/boundary files are the archival record;
    # selector/replay/function graph binaries are rebuildable.
    if spec.key == "clean_rebuild":
        if rel in {
            "OldHospital/clean_rebuild_results.json",
            "OldHospital/v11_alternating_results.json",
            "OldHospital/clean_input_boundary.json",
        }:
            return "KEEP", "clean rebuild 主结果与输入边界"
        if ext in {".pt", ".pth", ".pkl"}:
            return "DELETE_CANDIDATE", "可重建的 selector/replay/function-graph/map 二进制"
        if ext in {".json", ".jsonl", ".partial"}:
            return "ARCHIVE", "历史 replay/评估结果；可集中保存"
        if ext in {".png", ".jpg", ".jpeg"}:
            return "ARCHIVE", "历史诊断可视化"
        if ext in {".log", ".sh", ".yaml", ".yml", ".txt", ".pid"}:
            return "ARCHIVE", "历史命令、配置或日志"
        return "ARCHIVE", "clean rebuild 辅助文件"

    # Standalone caches: every file is a regenerable cache.
    if spec.key in {"rgb_prior_cache", "query_cache"}:
        return "DELETE_CANDIDATE", "独立 query cache，可从 prior/query 图像重新生成"

    # v6 feedback series: feedback/probe tensors and large raw diagnostics are
    # disposable; small JSON summaries are retained/archived.
    if spec.key in {"v6_feedback_core", "v6_feedback_control_20260824", "v6_feedback_control_20260825"}:
        if ext in {".pt", ".pth"}:
            return "DELETE_CANDIDATE", "v6 历史 feedback/probe/map 二进制，可重建"
        if ext == ".json":
            if allocated > 100 * 1024**2 or apparent > 100 * 1024**2:
                return "DELETE_CANDIDATE", "大体积 raw diagnostic JSON，可由历史 run 重建"
            return "ARCHIVE", "v6 feedback 小型 summary/provenance"
        return "ARCHIVE", "v6 feedback 历史辅助文件"

    return "ARCHIVE", "未匹配规则的历史文件，需人工确认"


def fmt_bytes(n: int) -> str:
    return f"{n / 1024**3:.2f} GiB"


def main() -> None:
    generated = datetime.now().astimezone().isoformat(timespec="seconds")
    all_rows = []
    summaries = []

    for spec in ROOTS:
        if not spec.path.exists():
            summaries.append((spec, 0, 0, 0, {}, {}))
            continue
        counts = defaultdict(lambda: [0, 0, 0])
        rollups = defaultdict(lambda: defaultdict(lambda: [0, 0, 0]))
        rows = []
        for path, apparent, allocated in file_rows(spec.path):
            action, reason = classify(spec, path, apparent, allocated)
            row = {
                "root": spec.key,
                "root_path": str(spec.path),
                "relative_path": path.relative_to(spec.path).as_posix(),
                "apparent_bytes": apparent,
                "allocated_bytes": allocated,
                "action": action,
                "reason": reason,
            }
            rows.append(row)
            c = counts[action]
            c[0] += 1
            c[1] += apparent
            c[2] += allocated
            top = path.relative_to(spec.path).parts[0]
            rc = rollups[top][action]
            rc[0] += 1
            rc[1] += apparent
            rc[2] += allocated
        all_rows.extend(rows)
        summaries.append((spec, len(rows), sum(r["apparent_bytes"] for r in rows), disk_bytes(spec.path), counts, rollups))

    OUT_TSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_TSV.open("w", encoding="utf-8") as f:
        f.write("root\troot_path\trelative_path\tapparent_bytes\tallocated_bytes\taction\treason\n")
        for r in sorted(all_rows, key=lambda x: (x["root"], x["relative_path"])):
            f.write(
                "\t".join(
                    str(r[k]).replace("\t", " ").replace("\n", " ")
                    for k in ("root", "root_path", "relative_path", "apparent_bytes", "allocated_bytes", "action", "reason")
                )
                + "\n"
            )

    lines = []
    lines.append("# STDLoc / LaFGS cleanup dry-run v2 (read-only)")
    lines.append("")
    lines.append(f"Generated: `{generated}`")
    lines.append("")
    lines.append("本报告只读取文件并生成清单；没有删除、移动或覆盖 `/mnt/pool/sqy` 下任何文件。`G4Splat_runs` 不在扫描范围内。")
    lines.append("")
    lines.append("当前运行保护：`/mnt/pool/sqy/anygsloc_execution_queue_20260909` 仍有 AnyGSLoc scene/map/evaluation 任务，因此本报告只把它作为外部保护项，不对其做逐文件清理。")
    lines.append("")
    lines.append("动作含义：`KEEP` 保留原位；`ARCHIVE` 只保留小型 provenance/结果或待集中归档；`DELETE_CANDIDATE` 表示可重建、但仍需用户最终批准后才删除。")
    lines.append("")
    lines.append("## Root summary")
    lines.append("")
    lines.append("| root | allocated (`du`) | files | KEEP | ARCHIVE | DELETE_CANDIDATE |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for spec, n, apparent, allocated, counts, rollups in summaries:
        def cell(action: str) -> str:
            c = counts.get(action, [0, 0, 0])
            return f"{c[0]} files / {fmt_bytes(c[2])}"
        lines.append(
            f"| `{spec.path}` | {fmt_bytes(allocated)} | {n} | {cell('KEEP')} | {cell('ARCHIVE')} | {cell('DELETE_CANDIDATE')} |"
        )

    lines.append("")
    lines.append("## Per-root interpretation")
    lines.append("")
    for spec, n, apparent, allocated, counts, rollups in summaries:
        lines.append(f"### `{spec.path}`")
        lines.append("")
        lines.append(f"- Scope: {spec.description}")
        lines.append(f"- Allocated size: **{fmt_bytes(allocated)}**; apparent file size: **{fmt_bytes(apparent)}**; files: **{n}**.")
        for action in ("KEEP", "ARCHIVE", "DELETE_CANDIDATE"):
            c = counts.get(action, [0, 0, 0])
            lines.append(f"- `{action}`: {c[0]} files, {fmt_bytes(c[2])} allocated.")
        lines.append(f"- Exact file-level manifest: [`stdloc_cleanup_dry_run_20260909_v2.tsv`](/root/STDLoc/docs/stdloc_cleanup_dry_run_20260909_v2.tsv), filtered by `root={spec.key}`.")
        lines.append("")

    lines.append("## First-level directory rollups")
    lines.append("")
    lines.append("下表按每个 root 的一级目录汇总；更深层的逐文件动作见 TSV。")
    lines.append("")
    for spec, n, apparent, allocated, counts, rollups in summaries:
        lines.append(f"### `{spec.path}`")
        lines.append("")
        lines.append("| first-level directory | allocated | KEEP | ARCHIVE | DELETE_CANDIDATE |")
        lines.append("|---|---:|---:|---:|---:|")
        for top, action_counts in sorted(
            rollups.items(),
            key=lambda item: sum(v[2] for v in item[1].values()),
            reverse=True,
        ):
            total_alloc = sum(v[2] for v in action_counts.values())

            def dcell(action: str) -> str:
                c = action_counts.get(action, [0, 0, 0])
                return f"{c[0]} / {fmt_bytes(c[2])}"

            lines.append(f"| `{top}` | {fmt_bytes(total_alloc)} | {dcell('KEEP')} | {dcell('ARCHIVE')} | {dcell('DELETE_CANDIDATE')} |")
        lines.append("")

    lines.append("## Additional large roots not included in the exact manifest")
    lines.append("")
    lines.append("这些目录在下一轮 dry-run 中优先复核；本报告不对它们做任何操作：")
    lines.append("")
    for path in ADDITIONAL_ROOTS:
        if path.exists():
            lines.append(f"- `{path}` — allocated **{fmt_bytes(disk_bytes(path))}**")
        else:
            lines.append(f"- `{path}` — not present")

    lines.append("")
    lines.append("## Protection notes")
    lines.append("")
    lines.append("- 保留当前 indoor 先验、5 个 Cambridge MAtCha 先验、Cambridge/12Scenes 原始或处理数据。")
    lines.append("- `anygsloc_paper_experiments_20260902` 的 PLY 先验不在该目录内；其中 feature cache/map 的候选删除不会直接删除先验 PLY。")
    lines.append("- 在用户批准前，本清单中的 `DELETE_CANDIDATE` 只表示候选，不执行删除。")
    lines.append("")
    lines.append("## Size accounting")
    lines.append("")
    lines.append("文件级 `apparent_bytes` 与 allocated bytes 都写入 TSV；总目录占用使用 `du -sB1`，因此可能因稀疏文件/目录块与文件 apparent size 略有差异。")
    lines.append("")

    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(OUT_MD)
    print(OUT_TSV)


if __name__ == "__main__":
    main()
