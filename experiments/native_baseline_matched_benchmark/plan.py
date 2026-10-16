"""CPU preparation and later all-assets binding. No GPU import or launch."""
import argparse
from pathlib import Path
from .common import *
from experiments.native_baseline_repair.query_plan import validate_contract as validate_accuracy, replace_value

DESIGN = {
 "schema": "native_any_matched_timing_design_v1", "seed": SEED, "namespace": NAMESPACE,
 "scenes": list(SCENES), "rounds": list(ROUNDS), "warmup_per_round": WARMUP,
 "measured_per_round": MEASURED, "cells": 30, "query_semantics": True,
 "selection": "SHA256(namespace NUL seed NUL scene NUL complete name); lowest100; execute frozen full-name order",
 "accuracy": "separate original all1918-query experiment; never replace accuracy with timing subset",
 "process": "one cell/round per fresh finite Runner job; max3000 seconds; no GPU work authorized by preparation",
 "features": "native feature extraction on every warmup and measured call; no query-feature cache",
 "native_core": "unchanged original wrapper synchronized localize; coarse prefix includes shared precomputation; dense increment is core minus coarse prefix",
 "native_frame": "reopen+decode frozen RGB, native camera materialization (including geometry and device work), native query bookkeeping/H2D, mask preprocessing, native localize; excludes post-localize accuracy metrics and serialization",
 "native_mask": "synchronized conversion/composition/device transfer/application together; pure mask compose unavailable",
 "native_decode": "PIL Image.open+load on every invocation; native SceneDataset __getitem__ consumes an equivalent fresh image",
 "STDLoc": "native SuperPoint/matching/PoseLib/dense1; labelled external semantic RGB preprocessing; core takes no GT; frame includes camera/GT metadata construction",
 "ULFLoc": "native SuperPoint/matching/PoseLib/dense5; native semantic branch; core and frame INCLUDE GT projection/inlier diagnostics",
 "AnyGSLoc": "same frozen Full/Compact maps, calibration and unchanged Localizer; decode/preprocess and CPU mask composition via existing dataset",
 "segmentation": "precomputed masks only; segmentation-network inference excluded, never assumed zero",
 "initialization": "fresh process; timer starts after adapter dependency imports; native wrapper setup or Any dataset/model/mask archive through first query; excludes interpreter/dependency import startup and launch hash verification",
 "native_storage": "sum exact unique model inference asset bytes: cfg_args+Gaussian PLY+STD detector/indices or ULF landmark/features; shared SuperPoint weights reported separately; sources/images/masks/training checkpoints excluded",
 "any_storage": "exact deployment_map.pt bytes; shared SuperPoint weights and offline Gaussian/cache assets excluded",
 "memory": "torch allocated/reserved peaks per frame incl pre-core captures, maxima over measured frames separately from warmup/init; process ru_maxrss is lifetime incl init; not whole-GPU VRAM",
 "isolation": "all training/accuracy/heavy CPU/I/O released by parent, one physical GPU UUID; common advisory lock, exclusive run dirs, 1s external GPU-process checks and host evidence; host quiet release attestation required",
 "unavailable": {"native_mask_compose_ms": None, "Any_coarse_prefix_ms": None, "Any_dense_increment_ms": None},
 "aggregation": "per-round100 query distributions; pool300 per cell; never include warmup; report three rounds and scene-equal aggregation separately",
}

def code_files():
    folder = Path(__file__).parent
    return (sorted(folder.glob("*.py")) + sorted((REPO / "tests").glob("test_native_baseline_matched_benchmark*.py"))
            + [REPO / "experiments/native_baseline_repair/query_report.py",
               REPO / "experiments/native_baseline_repair/query_results.py"])

def prepare(root=PREPARATION):
    root = Path(root)
    if (root / "protocol.json").exists():
        raise FileExistsError(root / "protocol.json")
    pending_path = ROOT / "timing_subset_pending_plan_v2.json"
    query_path = ROOT / "query_semantic_on/protocol_v2.json"
    any_path = ANY_ROOT / "protocol.json"
    for path, expected in ((pending_path, PENDING_SHA), (query_path, QUERY_SHA), (any_path, ANY_SHA)):
        if sha(path) != expected:
            raise ValueError("preapproved reference changed: " + str(path))
    pending, query, any_protocol = read(pending_path), read(query_path), read(any_path)
    acceptance_gate = accuracy_gate()
    selected = {}
    for scene in SCENES:
        selected[scene] = check_selection(scene, pending["scenes"][scene])
        if selected[scene]["full_query_order"] != query["scenes"][scene]["query_names"]:
            raise ValueError("selection detached from full accuracy names")
        qref = any_protocol["scenes"][scene]["reference_snapshot"]
        verify(qref)
        if sorted(q["name"] for q in read(qref["path"])["queries"]) != sorted(selected[scene]["full_query_order"]):
            raise ValueError("Any/native full query populations differ")
    value = dict(design=DESIGN, selection=selected,
                 accuracy_acceptance_gate=acceptance_gate,
                 pending_plan=ref(pending_path), accuracy_protocol=ref(query_path), any_protocol=ref(any_path),
                 code_sha256={str(p.resolve()): sha(p) for p in code_files()},
                 gpu_run_performed=False, status="design frozen; final30-cell asset binding and independent review required")
    write_new(root / "protocol.json", value)
    write_new(root / "preflight.json", dict(complete=True, protocol=ref(root / "protocol.json"),
              scene_query_counts={s: v["full_query_count"] for s, v in selected.items()},
              selected_per_scene=MEASURED, warmup_per_round=WARMUP, rounds=3,
              gpu_run_performed=False, final_assets_bound=False))
    return ref(root / "protocol.json")

def protocol(root):
    path = Path(root) / "protocol.json"
    value = read(path)
    if value["design"] != DESIGN:
        raise ValueError("timing design drift")
    expected = {str(p.resolve()): sha(p) for p in code_files()}
    if value["code_sha256"] != expected:
        raise ValueError("matched timing source/test drift")
    if value["accuracy_acceptance_gate"] != accuracy_gate():
        raise ValueError("full-query acceptance gate detached")
    for r in (value["pending_plan"], value["accuracy_protocol"], value["any_protocol"]):
        verify(r)
    for scene in SCENES:
        check_selection(scene, value["selection"][scene])
    pre = read(Path(root) / "preflight.json")
    if pre.get("complete") is not True or pre.get("gpu_run_performed") is not False or pre["protocol"] != ref(path):
        raise ValueError("CPU preparation receipt invalid")
    return value

def any_dependencies(p):
    refs = list(p["evidence"])
    for scene, item in p["scenes"].items():
        refs.append(item["reference_snapshot"])
        refs.extend(item["maps"].values())
    for path, digest in p["code_sha256"].items():
        full = Path(path) if Path(path).is_absolute() else REPO / path
        if sha(full) != digest:
            raise ValueError("old Any code changed: " + str(full))
        refs.append(ref(full))
    return refs

def accuracy_gate():
    path = ROOT / "query_semantic_on/report_cpu_independent_review.json"
    if sha(path) != "03ee5dbbdeea3bdfe25636cd9d508292b77f82dc9d302bb77404c6734d03f081":
        raise ValueError("approved full-query review changed")
    return dict(review=ref(path),
                api="experiments.native_baseline_repair.query_report.load_cell",
                arguments=["canonical contract parent", "frozen accuracy protocol", "canonical accuracy protocol path"],
                bind="all20 complete chains", round="this native cell complete chain")

def accepted_accuracy(path, qp, protocol_path):
    """Use the already-reviewed full-query gate, including terminal and receipt."""
    from experiments.native_baseline_repair import query_report
    if (Path(query_report.__file__).resolve() != REPO / "experiments/native_baseline_repair/query_report.py"
            or sha(query_report.__file__) != "69d6a224f988e1077bb34d85ded8ec095003ec1848e46a6b1b3096032a96141b"):
        raise ValueError("approved full-query acceptance code changed")
    return query_report.load_cell(path.parent, qp, protocol_path)

def authority_cells(p, accept_all=False, selected_case=None):
    """Rebuild identities from frozen protocols, never from assets/round metadata.

    All20 complete chains are checked at binding. Each native round rechecks its
    full chain. Other cells require canonical small references and actual storage
    sizes; do not rehash all30 models during every single-round preflight.
    """
    from experiments.native_baseline_repair.query_plan import expected_cell, asset_names
    pp = Path(p["accuracy_protocol"]["path"])
    qp, ap = read(pp), read(p["any_protocol"]["path"])
    expected_keys = {m + "_" + s + "_mapping_semantics_" + z + "_eval"
                     for s in SCENES for m in ("STDLoc", "ULFLoc") for z in ("off", "on")}
    if set(qp["cells"]) != expected_keys:
        raise ValueError("authority is not the original twenty-cell matrix; no partial binding")
    required = [pp.parent / "cells" / key / name for key in sorted(expected_keys)
                for name in ("contract.json", "verification.json")]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise ValueError("final assets not ready; no partial binding: " + ", ".join(missing))
    cells = {}
    for key in sorted(expected_keys):
        path = pp.parent / "cells" / key / "contract.json"
        if path.resolve() != path or (path.parent / "verification.json").resolve() != path.parent / "verification.json":
            raise ValueError("canonical accuracy source redirected")
        c = read(path)
        original = qp["cells"][key]
        expected = expected_cell(qp, key, pp)
        if c.get("case") != key or any(c.get(k) != v for k, v in expected.items()):
            raise ValueError("authority contract metadata differs from original recipe")
        model = Path(original["model_source"])
        if set(c["model_assets"]) != set(asset_names(original["method"])):
            raise ValueError("authority inference assets differ")
        size = 0
        for rel, value in c["model_assets"].items():
            actual = model / rel
            if value["path"] != str(actual) or value["bytes"] != actual.stat().st_size:
                raise ValueError("authority storage path/bytes differ")
            size += actual.stat().st_size
        accuracy_path = path.parent / "verification.json"
        accuracy = read(accuracy_path)
        results_path = Path(expected["native_output"]) / "results.json"
        if results_path.resolve() != results_path:
            raise ValueError("canonical accuracy results redirected")
        # Unselected cells use the canonical verification's frozen result reference
        # and actual size; their full bytes were checked at bind and are rechecked
        # by load_cell whenever that cell is selected.
        result_ref = accuracy["results"]
        if result_ref["path"] != str(results_path) or result_ref["bytes"] != results_path.stat().st_size:
            raise ValueError("canonical accuracy result path/bytes differ")
        if (accuracy.get("complete") is not True or accuracy.get("case") != key
                or accuracy.get("contract") != ref(path)
                or accuracy.get("query_names") != expected["query_names"]
                or accuracy.get("results") != result_ref):
            raise ValueError("full-query native accuracy reference detached")
        if accept_all or key == selected_case:
            accepted = accepted_accuracy(path, qp, pp)
            if (accepted["contract"] != ref(path) or accepted["verification"] != ref(accuracy_path)
                    or accepted["results"] != result_ref or accepted["query_names"] != expected["query_names"]
                    or accepted["deployment_bytes"] != size):
                raise ValueError("full-query accepted chain differs")
            if accept_all:
                for r in [*c["scene_inputs"].values(), c["query_snapshot"], c["native_entry"], c["wrapper"], c["native_config"]]:
                    verify(r)
        cells[key] = dict(family="native", method=original["method"], scene=original["scene"],
                          mapping_semantics=original["mapping_semantics"], accuracy_contract=ref(path),
                          accuracy_reference=ref(accuracy_path), accuracy_results=result_ref,
                          storage_bytes=size)
    for scene in SCENES:
        for variant in ("full", "compact"):
            mapref = ap["scenes"][scene]["maps"][variant]
            if Path(mapref["path"]).stat().st_size != mapref["bytes"]:
                raise ValueError("Any authoritative storage bytes differ")
            key = "AnyGSLoc_" + scene + "_" + variant
            accuracy_path = Path(p["any_protocol"]["path"]).parent / "runs" / scene / variant / "round0.json"
            if accuracy_path.resolve() != accuracy_path:
                raise ValueError("canonical Any accuracy source redirected")
            if accept_all or key == selected_case:
                accuracy = read(accuracy_path)
                if [q["image_name"] for q in accuracy["queries"]] != p["selection"][scene]["full_query_order"]:
                    raise ValueError("old Any full-query accuracy population differs")
            cells[key] = dict(family="any", method="AnyGSLoc", scene=scene, variant=variant,
                              accuracy_reference=ref(accuracy_path),
                              map=mapref, storage_bytes=mapref["bytes"])
    return cells

def bind(root=PREPARATION):
    """Only call after all20 full-query acceptances exist; hashes real assets."""
    root = Path(root)
    p = protocol(root)
    if (root / "assets.json").exists():
        raise FileExistsError(root / "assets.json")
    cells = authority_cells(p, accept_all=True)
    dependencies = any_dependencies(read(p["any_protocol"]["path"]))
    for r in dependencies:
        verify(r)
    assets = dict(schema="matched_timing_assets_v1", protocol=ref(root / "protocol.json"), cells=cells,
                  any_dependencies=dependencies, complete=True)
    write_new(root / "assets.json", assets)
    for key, cell in cells.items():
        for round_id in ROUNDS:
            write_new(root / "contracts" / key / ("round%d.json" % round_id),
                      dict(schema="matched_timing_round_v1", case=key, round=round_id,
                           protocol=ref(root / "protocol.json"), assets=ref(root / "assets.json"),
                           cell=cell, selection=p["selection"][cell["scene"]],
                           output=str((root / "runs" / key / ("round%d" % round_id)).resolve()),
                           runner_timeout_seconds=3000))
    return ref(root / "assets.json")

def validate_round(c, root):
    p = protocol(root)
    if c.get("schema") != "matched_timing_round_v1" or c["protocol"] != ref(Path(root) / "protocol.json"):
        raise ValueError("round protocol identity invalid")
    if c["assets"] != ref(Path(root) / "assets.json"):
        raise ValueError("assets detached")
    a = read(c["assets"]["path"])
    authoritative = dict(schema="matched_timing_assets_v1", complete=True, protocol=c["protocol"],
                         cells=authority_cells(p, selected_case=c["case"]),
                         any_dependencies=any_dependencies(read(p["any_protocol"]["path"])))
    if a != authoritative:
        raise ValueError("assets differ from authoritative original protocols/accepted sources")
    if c["case"] not in a["cells"] or c["cell"] != a["cells"][c["case"]] or c["selection"] != p["selection"][c["cell"]["scene"]]:
        raise ValueError("round cell or selection altered")
    schedule(c["selection"], c["round"])
    expected = Path(root) / "runs" / c["case"] / ("round%d" % c["round"])
    if c["output"] != str(expected.resolve()) or c["runner_timeout_seconds"] != 3000:
        raise ValueError("round destination/budget changed")
    return p

def native_contract(c):
    """Derive only new output/model view routing; no numerical arguments change."""
    base = read(c["cell"]["accuracy_contract"]["path"])
    out = Path(c["output"])
    new = dict(base)
    new["model_view"] = str(out / "model")
    new["receipt"] = str(out / "native_receipt.json")
    new["native_output"] = str(out / "native_output" if base["method"] == "STDLoc" else out / "model/matched_timing")
    argv = list(base["wrapper_argv"])
    for flag, value in (("--receipt", new["receipt"]), ("-m", new["model_view"]), ("--run-tag", "matched_timing")):
        argv = replace_value(argv, flag, value)
    if "--warmup" in argv:
        argv = replace_value(argv, "--warmup", "0")
    else:
        pos = argv.index("--")
        argv[pos:pos] = ["--warmup", "0"]
    new["wrapper_argv"] = argv
    return new

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "bind"))
    parser.add_argument("--root", type=Path, default=PREPARATION)
    args = parser.parse_args()
    print(__import__("json").dumps((prepare if args.action == "prepare" else bind)(args.root)))

if __name__ == "__main__":
    main()
