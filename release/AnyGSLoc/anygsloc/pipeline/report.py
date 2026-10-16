"""Scene-equal localization statistics; missing scenes never become a full result."""

from pathlib import Path

import numpy as np

from anygsloc.pipeline.io import read_json, write_json


def metrics(rows):
    if not rows:
        raise ValueError("empty result")

    def errors(key):
        return np.asarray(
            [
                float(r[key])
                if r.get("success", True) and r.get(key) is not None
                else float("inf")
                for r in rows
            ]
        )

    te, re = errors("translation_error_cm"), errors("rotation_error_deg")

    def percentile(values, fraction):
        values = np.sort(values)
        index = (len(values) - 1) * fraction
        low, high = int(np.floor(index)), int(np.ceil(index))
        if high == low or not np.isfinite(values[high]):
            return float(values[high])
        return float(values[low] + (values[high] - values[low]) * (index - low))

    result = {
        "queries": len(rows),
        "failures": sum(not r.get("success", True) for r in rows),
        "median_TE_cm": float(np.median(te)),
        "median_RE_deg": float(np.median(re)),
        "mean_TE_cm": float(te.mean()),
        "mean_RE_deg": float(re.mean()),
        "P90_TE_cm": percentile(te, 0.9),
        "P95_TE_cm": percentile(te, 0.95),
        "R5_pct": float(100 * np.mean((te < 5) & (re < 5))),
        "R15_pct": float(100 * np.mean((te < 15) & (re < 5))),
    }
    return result


def aggregate(per_scene):
    if not per_scene:
        raise ValueError("no scenes")
    keys = set.intersection(*(set(v) for v in per_scene.values()))
    sums = {"queries", "failures", "points", "bytes"}
    return {
        key: sum(v[key] for v in per_scene.values())
        if key in sums
        else float(np.mean([v[key] for v in per_scene.values()]))
        for key in sorted(keys)
    }


def safe(value):
    if isinstance(value, dict):
        return {k: safe(v) for k, v in value.items()}
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def report(config, scenes):
    import torch

    from anygsloc.artifacts import sha256

    root = Path(config["output"])
    groups = {}
    for scene in scenes:
        for path in (root / scene).glob("*/*/evaluations/*/results.json"):
            key = (
                "/".join(path.relative_to(root / scene).parts[:2])
                + "/"
                + path.parent.name
            )
            rows = read_json(path)
            manifest = read_json(path.parent / "manifest.json")
            if [r["image_name"] for r in rows] != manifest["queries"]:
                raise ValueError(f"incomplete or reordered results: {path}")
            if len(set(manifest["queries"])) != len(rows):
                raise ValueError(f"duplicate queries: {path}")
            expected = config["scenes"][scene].get("test_count")
            if expected and len(rows) != expected:
                raise ValueError(f"unexpected query count: {scene}")
            map_name = path.parent.name
            cell = path.parents[2]
            if map_name in ("center", "depth", "ray"):
                map_path = cell / "geometry" / map_name / "deployment_map.pt"
            elif map_name in ("full", "single"):
                map_path = cell / f"{map_name}.pt"
            else:
                map_path = cell / "selection" / f"{map_name}.pt"
            if not map_path.exists() and config["scenes"][scene].get(f"{map_name}_map"):
                map_path = Path(config["scenes"][scene][f"{map_name}_map"])
            if sha256(map_path) != manifest["map_sha256"]:
                raise ValueError(f"evaluated map changed: {map_path}")
            value = metrics(rows)
            value.update(
                points=len(
                    torch.load(map_path, map_location="cpu", weights_only=True)[
                        "anchor_xyz"
                    ]
                ),
                bytes=map_path.stat().st_size,
            )
            groups.setdefault(key, {})[scene] = value
    result = {}
    for name, values in groups.items():
        if set(values) != set(scenes):
            raise ValueError(f"missing scenes for {name}: {set(scenes) - set(values)}")
        result[name] = {"per_scene": values, "scene_equal": aggregate(values)}
    if not result:
        raise ValueError("no completed evaluations")
    write_json(
        root / "report.json",
        safe(
            {
                "results": result,
                "scene_order": scenes,
                "complete_configured_dataset": set(scenes) == set(config["scenes"]),
            }
        ),
    )
