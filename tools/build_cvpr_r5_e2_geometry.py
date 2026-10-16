"""R5 E2 caption clarification; retain the accepted graphic byte-for-byte."""
from pathlib import Path
import hashlib
import json
import shutil

ROOT = Path(__file__).resolve().parents[1]
PREV = ROOT / "paper/experiments_cvpr_r4_20261003"
OUT = ROOT / "paper/experiments_cvpr_r5_20261003"
STEM = "E2_geometry_source_evidence"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    source = PREV / f"{STEM}.provenance.json"
    p = json.loads(source.read_text())
    for ext in ["pdf", "svg", "png"]:
        old = PREV / f"{STEM}.{ext}"
        new = OUT / old.name
        assert sha(old) == p["outputs"][STEM][ext]["sha256"]
        shutil.copy2(old, new)
        assert sha(new) == sha(old)
        p["outputs"][STEM][ext]["path"] = str(new)
    geometry_table = ROOT / "paper/AnyGSLoc_TRO_v6/tables/geometry.tex"
    table_text = geometry_table.read_text()
    for expected in [r"\label{tab:geometry}", "11.667/0.157", "9.911/0.160", "8.765/0.133"]:
        assert expected in table_text
    p["revision"] = 5
    p["previous_figure_provenance"] = {"path": str(source), "sha256": sha(source)}
    p["source_script"] = {"path": str(Path(__file__).resolve()), "sha256": sha(__file__)}
    p["r5_changes"] = ["Clarified that these are mapping-view residuals and linked to the separate, existing five-scene query-localization control."]
    p["r5_graphic_byte_identical_to_r4"] = True
    p["r5_data_unchanged"] = True
    p["geometry_ablation_reference"] = {"path": str(geometry_table), "sha256": sha(geometry_table), "label": "tab:geometry",
        "mean_of_scene_median_translation_error_cm": {"center": 11.667, "depth": 9.911, "ray": 8.765},
        "relationship": "Separate existing five-scene query-localization evidence, not a generalization conclusion derived from the single-scene ECDF."}
    caption = (
        "Mapping-view reprojection residuals in St. Mary's Church. The ECDF evaluates the same 1,289,079 archived observations of 55,817 common landmarks across 1,465 contributing views (1,487 archived views). "
        "Landmark IDs and descriptors are fixed; only XYZ changes. Each source is projected into the same cameras and compared with the same detected features. "
        "Invalid or behind-camera projections count as infinite residuals; observations outside the displayed range remain in the denominator. "
        "Because ray coordinates were estimated from these observations, this plot measures mapping fit. Query-localization accuracy is evaluated separately in the existing five-scene coordinate-source ablation: the mean of scene-median translation errors is 11.667, 9.911, and 8.765 cm for center, depth, and ray coordinates, respectively."
    )
    (OUT / f"{STEM}.caption.txt").write_text(caption + "\n\nProtocol and limitations:\n" + "\n".join("- " + s for s in p["limitations"]) + "\n")
    (OUT / f"{STEM}.provenance.json").write_text(json.dumps(p, indent=2) + "\n")
    print("E2 PDF, SVG, and PNG unchanged from R4; caption and provenance updated.")


if __name__ == "__main__":
    main()
