"""Carry forward the verified E2 curve and audit its archived observation counts.

No geometry, matching, or localization is run. The PDF/SVG/PNG are byte-identical
to R2; only caption/provenance metadata distinguish the complete camera archive
from the views contributing observations to this paired residual population.
"""
from pathlib import Path
import hashlib
import json
import shutil

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "paper/experiments_cvpr_r2_20261003"
OUT = ROOT / "paper/experiments_cvpr_r3_20261003"
STEM = "E2_geometry_source_evidence"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    source_provenance = SOURCE / f"{STEM}.provenance.json"
    p = json.loads(source_provenance.read_text())
    residual_path = Path(p["residual_asset"]["path"])
    assert sha(residual_path) == p["residual_asset"]["sha256"]
    data = np.load(residual_path)
    counts = {key: int(len(data[key])) for key in data.files}
    assert len(set(counts.values())) == 1
    observation_count = counts["ray"]
    landmark_count = int(np.unique(data["anchor_rows"]).size)
    view_ids = np.unique(data["query_indices"])
    contributing_views = int(view_ids.size)
    archive_views = int(p["population_mapping_views"])
    assert 0 <= int(view_ids.min()) <= int(view_ids.max()) < archive_views
    assert observation_count == p["population_observations"]
    assert landmark_count == p["population_landmarks"]
    checks = {}
    for key in ("center", "depth", "ray"):
        v = data[key]
        actual = {
            "N": int(v.size),
            "finite": int(np.isfinite(v).sum()),
            "infinite": int(np.isinf(v).sum()),
            "nan": int(np.isnan(v).sum()),
            "median_px": float(np.median(v)),
        }
        for field in ("N", "finite", "median_px"):
            assert actual[field] == p["statistics"][key][field]
        assert actual["nan"] == 0
        checks[key] = actual
    for ext in ("pdf", "svg", "png"):
        source = SOURCE / f"{STEM}.{ext}"
        target = OUT / source.name
        expected = p["outputs"][STEM][ext]["sha256"]
        assert sha(source) == expected
        shutil.copy2(source, target)
        assert sha(target) == expected
        p["outputs"][STEM][ext]["path"] = str(target)
    p["population_mapping_views"] = contributing_views
    p["archive_mapping_views"] = archive_views
    p["contributing_mapping_views"] = contributing_views
    p["archive_views_without_common_track_observations"] = archive_views - contributing_views
    p["r3_count_correction"] = (
        "R2 population_mapping_views counted all archived camera names. R3 counts "
        "the unique query_indices actually present in the complete common-track CSR. "
        "The residual arrays, evaluated observation population, statistics, and curve "
        "are unchanged."
    )
    p["carried_forward_from"] = {
        "provenance_path": str(source_provenance),
        "provenance_sha256": sha(source_provenance),
        "figure_bytes_unchanged": True,
    }
    caption = (SOURCE / f"{STEM}.caption.txt").read_text()
    old = "across 1,487 mapping views."
    new = (
        f"across {contributing_views:,} contributing mapping views "
        f"(from an archive of {archive_views:,} views)."
    )
    assert old in caption
    caption = caption.replace(old, new)
    (OUT / f"{STEM}.caption.txt").write_text(caption)
    (OUT / f"{STEM}.provenance.json").write_text(json.dumps(p, indent=2) + "\n")
    audit = {
        "figure": STEM,
        "audit_scope": "Read-only verification of existing arrays and source script; copy and metadata correction only.",
        "source_script": str(ROOT / "tools/build_cvpr_r2_e2_geometry.py"),
        "source_script_sha256": sha(ROOT / "tools/build_cvpr_r2_e2_geometry.py"),
        "residual_asset": p["residual_asset"],
        "reprojection_error_formula": "e_ij = || pi(K_i (R_i X_j + t_i)) - (0.5, 0.5) - u_ij ||_2",
        "pi_definition": "Perspective division of the first two homogeneous image coordinates by the third.",
        "units": "Pixels in each archived camera's native feature-image grid; not resized figure pixels.",
        "comparison": "Identical landmark IDs and descriptors, paired archived cameras and detected 2D features; only frozen XYZ source changes.",
        "evaluation_type": "Mapping-fit diagnostic. Ray coordinates were estimated from these same mapping observations; not held-out query accuracy.",
        "population_observations": observation_count,
        "population_landmarks": landmark_count,
        "contributing_mapping_views": contributing_views,
        "archive_mapping_views": archive_views,
        "empty_archive_views": archive_views - contributing_views,
        "array_checks": checks,
        "ecdf_denominator": "Every observation, including behind-camera/nonfinite projections represented as +infinity; displayed range is not renormalized.",
        "x_axis": "Logarithmic, 0.05 to 1000 pixels; horizontal extent does not discard residuals from the ECDF denominator.",
        "limitations": p["limitations"],
        "figure_bytes_unchanged": True,
    }
    (OUT / "E2_independent_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps({"N": observation_count, "landmarks": landmark_count,
                      "contributing_views": contributing_views,
                      "archive_views": archive_views, "figure_bytes_unchanged": True}))


if __name__ == "__main__":
    main()
