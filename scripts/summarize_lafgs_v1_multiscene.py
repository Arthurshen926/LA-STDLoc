#!/usr/bin/env python3
"""Aggregate frozen LaFGS-v1.0 scene results and paired query bootstraps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml


LABELS = (
    "A0_bootstrap",
    "A1_reconstructed",
    "A2_family_all",
    "A3_p1_fixed512",
)
COMPARISONS = (
    ("A1_vs_A0", "A0_bootstrap", "A1_reconstructed"),
    ("A3_vs_A2", "A2_family_all", "A3_p1_fixed512"),
)


def _load_result(pointer: Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    result_dir = Path(pointer.read_text().strip())
    rows = json.loads((result_dir / "results.json").read_text())
    return (
        [str(row["image_name"]) for row in rows],
        np.asarray([float(row["sparse_TE"]) for row in rows]),
        np.asarray([float(row["sparse_AE"]) for row in rows]),
    )


def _seed_averaged_queries(
    scene_root: Path,
    label: str,
) -> tuple[list[str], np.ndarray, np.ndarray, list[int]]:
    names = None
    translations = []
    rotations = []
    seeds = []
    for pointer in sorted(
        (scene_root / "evaluation" / label).glob("seed*/result.path")
    ):
        current_names, te, ae = _load_result(pointer)
        if names is None:
            names = current_names
        elif current_names != names:
            raise ValueError(
                f"{scene_root.name}/{label} query ordering differs by seed"
            )
        translations.append(te)
        rotations.append(ae)
        seeds.append(int(pointer.parent.name.removeprefix("seed")))
    if not translations or names is None:
        raise FileNotFoundError(
            f"no completed evaluations for {scene_root.name}/{label}"
        )
    return (
        names,
        np.mean(np.stack(translations), axis=0),
        np.mean(np.stack(rotations), axis=0),
        seeds,
    )


def _metric_vector(values: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            np.median(values),
            np.mean(values),
            np.percentile(values, 90),
            np.mean(values <= 2.0) * 100.0,
            np.mean(values <= 5.0) * 100.0,
        ]
    )


def _paired_bootstrap(
    baseline: np.ndarray,
    candidate: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> dict:
    if baseline.shape != candidate.shape:
        raise ValueError("paired bootstrap inputs differ in shape")
    rng = np.random.default_rng(int(seed))
    count = len(baseline)
    draws = np.empty((int(samples), 5), dtype=np.float64)
    for offset in range(0, int(samples), 256):
        batch = min(256, int(samples) - offset)
        indices = rng.integers(0, count, size=(batch, count))
        for row, selection in enumerate(indices):
            draws[offset + row] = (
                _metric_vector(candidate[selection])
                - _metric_vector(baseline[selection])
            )
    point = _metric_vector(candidate) - _metric_vector(baseline)
    names = ("median_te_cm", "mean_te_cm", "p90_te_cm", "r2_pp", "r5_pp")
    lower, upper = np.percentile(draws, [2.5, 97.5], axis=0)
    return {
        name: {
            "delta": float(point[index]),
            "ci95": [float(lower[index]), float(upper[index])],
            "improvement_probability": float(
                np.mean(
                    draws[:, index] < 0
                    if index < 3
                    else draws[:, index] > 0
                )
            ),
        }
        for index, name in enumerate(names)
    }


def _aggregate_scene(scene_root: Path, samples: int, seed: int) -> dict:
    frozen = json.loads((scene_root / "frozen_results.json").read_text())
    config = yaml.safe_load(Path(frozen["canonical_config"]).read_text())
    result = {
        "bootstrap_anchor_count": int(
            config["initialization"]["scaffold_budget"]
        ),
        "anchor_count": frozen.get("anchor_count"),
        "map_bytes": frozen.get("map_bytes"),
        "deployment_total_bytes": frozen.get("deployment_total_bytes"),
        "results": frozen["results"],
        "paired_bootstrap": {},
    }
    cached = {}
    for label in LABELS:
        cached[label] = _seed_averaged_queries(scene_root, label)
    for name, baseline_label, candidate_label in COMPARISONS:
        baseline_names, baseline_te, _, baseline_seeds = cached[
            baseline_label
        ]
        candidate_names, candidate_te, _, candidate_seeds = cached[
            candidate_label
        ]
        if baseline_names != candidate_names:
            raise ValueError(
                f"{scene_root.name}/{name} query identities differ"
            )
        if baseline_seeds != candidate_seeds:
            raise ValueError(
                f"{scene_root.name}/{name} seed sets differ"
            )
        result["paired_bootstrap"][name] = {
            "query_count": len(baseline_names),
            "seeds": baseline_seeds,
            "translation": _paired_bootstrap(
                baseline_te,
                candidate_te,
                samples=samples,
                seed=seed,
            ),
        }
    return result


def _markdown(report: dict) -> str:
    lines = [
        "# LaFGS-v1.0 Frozen Cambridge Results",
        "",
        "| Scene | Variant | Anchors | Median TE | Mean TE | P90 TE | R5 | "
        "Inlier ratio | Hypotheses | Total ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for scene, payload in report["scenes"].items():
        for label in LABELS:
            aggregate = payload["results"][label]["seed_aggregate"]
            anchor_count = (
                payload["bootstrap_anchor_count"]
                if label == "A0_bootstrap"
                else payload.get("anchor_count", "")
            )

            def mean(name: str) -> float:
                return float(aggregate[name]["mean"])

            lines.append(
                f"| {scene} | {label} | {anchor_count} | "
                f"{mean('median_te_cm'):.3f} | {mean('mean_te_cm'):.3f} | "
                f"{mean('p90_te_cm'):.3f} | "
                f"{mean('recall_5cm_5deg_percent'):.2f}% | "
                f"{mean('solver_inlier_ratio_percent'):.2f}% | "
                f"{mean('mean_hypotheses'):.0f} | {mean('total_ms'):.1f} |"
            )
    lines.extend(
        [
            "",
            "Paired bootstrap deltas use the per-query mean over solver seeds; "
            "negative TE and positive recall deltas are improvements.",
            "",
            "| Scene | Comparison | Delta median | 95% CI | Delta mean | "
            "95% CI | Delta P90 | 95% CI | Delta R5 |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for scene, payload in report["scenes"].items():
        for name, comparison in payload["paired_bootstrap"].items():
            values = comparison["translation"]
            median = values["median_te_cm"]
            mean = values["mean_te_cm"]
            p90 = values["p90_te_cm"]
            r5 = values["r5_pp"]
            lines.append(
                f"| {scene} | {name} | {median['delta']:.3f} | "
                f"[{median['ci95'][0]:.3f}, {median['ci95'][1]:.3f}] | "
                f"{mean['delta']:.3f} | "
                f"[{mean['ci95'][0]:.3f}, {mean['ci95'][1]:.3f}] | "
                f"{p90['delta']:.3f} | "
                f"[{p90['ci95'][0]:.3f}, {p90['ci95'][1]:.3f}] | "
                f"{r5['delta']:+.2f} pp |"
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--scenes", nargs="+", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-markdown", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    report = {
        "schema": "lafgs_v1_frozen_multiscene_aggregate",
        "bootstrap_samples": int(args.bootstrap_samples),
        "bootstrap_seed": int(args.seed),
        "scenes": {},
    }
    for scene in args.scenes:
        scene_root = root / scene
        report["scenes"][scene] = _aggregate_scene(
            scene_root,
            samples=int(args.bootstrap_samples),
            seed=int(args.seed),
        )
    output_json = Path(args.output_json)
    output_markdown = Path(args.output_markdown)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_markdown.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    output_markdown.write_text(_markdown(report))
    print(output_json)
    print(output_markdown)


if __name__ == "__main__":
    main()
