"""Authority-binding regressions; fixtures never import or initialize CUDA."""
import copy
import json
from pathlib import Path
import pytest
from experiments.native_baseline_matched_benchmark import plan
from experiments.native_baseline_matched_benchmark.common import ref, read, selection, SCENES
from experiments.native_baseline_repair import query_report, query_results, query_plan

IDENTITY = [[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]]

def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return ref(path)

@pytest.fixture
def authority(tmp_path, monkeypatch):
    """Real load_cell chain, with only the already-tested native recipe gate isolated."""
    names = ["seq/frame%04d.png" % i for i in range(100)]
    qroot, aroot, root = tmp_path / "accuracy", tmp_path / "any", tmp_path / "matched"
    dummy = tmp_path / "dependency"
    dummy.write_text("unchanged")
    qp = dict(cells={}, scenes={})
    for scene in SCENES:
        qp["scenes"][scene] = dict(source=str(tmp_path / "data" / scene), query_names=names)
        for method in ("STDLoc", "ULFLoc"):
            for semantic in ("off", "on"):
                key = method + "_" + scene + "_mapping_semantics_" + semantic + "_eval"
                qp["cells"][key] = dict(method=method, scene=scene, mapping_semantics=semantic,
                    model_source=str(tmp_path / "models" / key),
                    original_job=dict(env={}, command=["python", str(dummy),
                        "--source", "data", "--receipt", "receipt", "--repo", str(tmp_path),
                        "--cfg", str(dummy), "--", "-s", "data", "-m", "model", "--images", "images"]))
    pp = qroot / "protocol.json"
    put(pp, qp)
    snapshot = tmp_path / "snapshot.json"
    put(snapshot, dict(queries=[dict(name=n, pose_w2c=IDENTITY) for n in names]))
    rows = [dict(gt_pose_w2c=IDENTITY, sparse=dict(pose_w2c=IDENTITY),
                 dense=[dict(pose_w2c=IDENTITY)]) for _ in names]
    metric = query_results.metrics([0.] * 100, [0.] * 100)
    for key, spec in qp["cells"].items():
        folder = qroot / "cells" / key
        expected = query_plan.expected_cell(qp, key, pp)
        assets = {}
        for rel in query_plan.asset_names(spec["method"]):
            asset = Path(spec["model_source"]) / rel
            asset.parent.mkdir(parents=True, exist_ok=True)
            asset.write_bytes(b"model")
            assets[rel] = ref(asset)
        c = dict(expected, case=key, protocol=ref(pp), model_assets=assets,
                 query_snapshot=ref(snapshot), scene_inputs={}, native_entry=ref(dummy),
                 wrapper=ref(dummy), native_config=ref(dummy))
        contract = put(folder / "contract.json", c)
        results = put(Path(c["native_output"]) / "results.json", rows)
        receipt = put(Path(c["receipt"]), dict(status="complete", method=c["method"], phase="evaluate",
            mapping_semantics=c["mapping_semantics"], query_semantics=True, seed=2026, warmup_queries=0,
            native_output=c["native_output"], timings=[{} for _ in names]))
        terminal = put(folder / "terminal.json", dict(terminal=True, exit_code=0, case=key,
            job_id="wc_job_fixture", evidence_source="WebCodex observe_jobs", contract_sha256=contract["sha256"]))
        launch = put(folder / "launch/launch.json", dict(contract_sha256=contract["sha256"]))
        log = folder / "launch/native.log"
        log.write_text("\n".join("Localize image: " + n for n in names))
        put(folder / "verification.json", dict(complete=True, case=key, contract=contract,
            query_names=names, metrics=dict(sparse=metric, dense=metric), results=results,
            native_receipt=receipt, terminal_evidence=terminal, launch=launch, native_log=ref(log),
            verifier=ref(query_results.__file__)))
    ap = dict(evidence=[], code_sha256={}, scenes={})
    for scene in SCENES:
        maps = {}
        for variant in ("full", "compact"):
            map_path = aroot / "maps" / scene / variant
            map_path.parent.mkdir(parents=True, exist_ok=True)
            map_path.write_bytes(variant.encode())
            maps[variant] = ref(map_path)
            put(aroot / "runs" / scene / variant / "round0.json",
                dict(queries=[dict(image_name=n, success=True, pose_w2c=IDENTITY) for n in names]))
        ap["scenes"][scene] = dict(maps=maps, reference_snapshot=ref(snapshot))
    put(aroot / "protocol.json", ap)
    p = dict(accuracy_protocol=ref(pp), any_protocol=ref(aroot / "protocol.json"),
             selection={s: selection(s, names) for s in SCENES})
    put(root / "protocol.json", p)
    monkeypatch.setattr(plan, "protocol", lambda root: p)
    # Test targets are caller admission and authoritative reconstruction, not
    # native recipe/training acceptance (covered by existing frozen tests).
    monkeypatch.setattr(query_report, "validate_contract", lambda *args: None)
    return root, p, qroot

def legacy_native_gate(folder):
    """Exact v3 thin acceptance condition; documents concrete previously accepted failures."""
    c = read(folder / "contract.json")
    v = read(folder / "verification.json")
    assert v.get("complete") is True and v.get("case") == c["case"] and v.get("contract") == ref(folder / "contract.json")
    assert v.get("query_names") == c["query_names"]
    assert v["results"]["path"] == str(Path(c["native_output"]) / "results.json")
    plan.verify(v["results"])

@pytest.mark.parametrize("fault", ["terminal_failure", "terminal_contract", "launch_contract",
                                   "receipt_semantics", "wrong_verifier", "wrong_log", "wrong_gt"])
def test_bind_rejects_fake_complete_through_real_acceptance_chain(authority, fault):
    root, p, qroot = authority
    key = sorted(read(p["accuracy_protocol"]["path"])["cells"])[0]
    folder = qroot / "cells" / key
    vpath = folder / "verification.json"
    v = read(vpath)
    mapping = dict(terminal_failure=("terminal_evidence", "exit_code", 1),
                   terminal_contract=("terminal_evidence", "contract_sha256", "wrong"),
                   launch_contract=("launch", "contract_sha256", "wrong"),
                   receipt_semantics=("native_receipt", "query_semantics", False))
    if fault in mapping:
        label, field, value = mapping[fault]
        path = Path(v[label]["path"])
        data = read(path); data[field] = value
        v[label] = put(path, data)
    elif fault == "wrong_verifier":
        v["verifier"] = dict(v["verifier"], sha256="bad")
    elif fault == "wrong_log":
        path = Path(v["native_log"]["path"])
        path.write_text(path.read_text().replace("seq/frame0000.png", "other.png"))
        v["native_log"] = ref(path)
    else:
        path = Path(v["results"]["path"])
        rows = read(path); rows[0]["gt_pose_w2c"] = copy.deepcopy(IDENTITY); rows[0]["gt_pose_w2c"][0][3] = 1
        v["results"] = put(path, rows)
    put(vpath, v)
    legacy_native_gate(folder)  # This is the concrete v3 failure path.
    with pytest.raises(ValueError):
        plan.bind(root)
    assert not (root / "assets.json").exists() and not (root / "contracts").exists()

def test_bind_all20_then_round_rechecks_only_selected_complete_chain(authority, monkeypatch):
    root, p, qroot = authority
    actual = plan.accepted_accuracy
    calls = []
    def observe(path, qp, pp):
        calls.append(path.parent.name)
        return actual(path, qp, pp)
    monkeypatch.setattr(plan, "accepted_accuracy", observe)
    plan.bind(root)
    assert len(calls) == len(set(calls)) == 20
    assert len(list((root / "contracts").glob("*/round*.json"))) == 90
    calls.clear()
    key = sorted(read(p["accuracy_protocol"]["path"])["cells"])[0]
    c = read(root / "contracts" / key / "round0.json")
    plan.validate_round(c, root)
    assert calls == [key]

def round_fixture(authority):
    root, p, qroot = authority
    cells = plan.authority_cells(p)
    assets = dict(schema="matched_timing_assets_v1", complete=True, protocol=ref(root / "protocol.json"),
                  cells=cells, any_dependencies=plan.any_dependencies(read(p["any_protocol"]["path"])))
    put(root / "assets.json", assets)
    key = sorted(cells)[0]
    cell = cells[key]
    c = dict(schema="matched_timing_round_v1", case=key, round=0, protocol=ref(root / "protocol.json"),
             assets=ref(root / "assets.json"), cell=copy.deepcopy(cell), selection=p["selection"][cell["scene"]],
             output=str(root / "runs" / key / "round0"), runner_timeout_seconds=3000)
    return root, p, assets, c

@pytest.mark.parametrize("fault", ["method", "scene", "semantics", "variant", "swap_map",
    "wrongpath_ref", "accuracy_results_path", "storage", "duplicate_cell", "missing_scene", "dependencies"])
def test_assets_and_round_resealed_together_still_reject(authority, fault):
    root, p, a, c = round_fixture(authority)
    plan.validate_round(c, root)
    native = sorted(k for k in a["cells"] if k.startswith("STDLoc"))[0]
    anykey = c["case"]
    if fault in ("method", "scene", "semantics"):
        field = dict(method="method", scene="scene", semantics="mapping_semantics")[fault]
        a["cells"][native][field] = dict(method="ULFLoc", scene="ShopFacade", semantics="on")[fault]
    elif fault == "variant":
        a["cells"][anykey]["variant"] = "full" if a["cells"][anykey]["variant"] == "compact" else "compact"
    elif fault == "swap_map":
        other = anykey.replace("_compact", "_full")
        a["cells"][anykey]["map"] = copy.deepcopy(a["cells"][other]["map"])
    elif fault in ("wrongpath_ref", "accuracy_results_path"):
        label = "accuracy_reference" if fault == "wrongpath_ref" else "accuracy_results"
        key = anykey if fault == "wrongpath_ref" else native
        original = Path(a["cells"][key][label]["path"])
        other = root / ("same_bytes_" + fault + ".json")
        other.write_bytes(original.read_bytes())
        a["cells"][key][label] = ref(other)
    elif fault == "storage":
        a["cells"][native]["storage_bytes"] += 1
    elif fault == "duplicate_cell":
        a["cells"][native + "_duplicate"] = a["cells"].pop(native)
    elif fault == "missing_scene":
        for key in list(a["cells"]):
            if a["cells"][key]["scene"] == "StMarysChurch":
                del a["cells"][key]
    else:
        a["any_dependencies"] = []
    put(root / "assets.json", a)
    c["assets"] = ref(root / "assets.json")
    c["cell"] = copy.deepcopy(a["cells"][c["case"]])
    # v3 would accept this refresh for all cases retaining exactly30 entries.
    if len(a["cells"]) == 30:
        assert c["cell"] == a["cells"][c["case"]] and c["assets"] == ref(root / "assets.json")
    with pytest.raises(ValueError):
        plan.validate_round(c, root)

def test_selected_round_rechecks_terminal_after_all_metadata_resealed(authority):
    root, p, a, c = round_fixture(authority)
    key = sorted(k for k in a["cells"] if k.startswith("STDLoc"))[0]
    cell = a["cells"][key]
    c.update(case=key, cell=copy.deepcopy(cell), selection=p["selection"][cell["scene"]],
             output=str(root / "runs" / key / "round0"))
    folder = Path(cell["accuracy_contract"]["path"]).parent
    terminal = read(folder / "terminal.json"); terminal["exit_code"] = 1
    put(folder / "terminal.json", terminal)
    v = read(folder / "verification.json"); v["terminal_evidence"] = ref(folder / "terminal.json")
    put(folder / "verification.json", v)
    a["cells"][key]["accuracy_reference"] = ref(folder / "verification.json")
    put(root / "assets.json", a)
    c.update(assets=ref(root / "assets.json"), cell=copy.deepcopy(a["cells"][key]))
    legacy_native_gate(folder)
    with pytest.raises(ValueError):
        plan.validate_round(c, root)

def test_full_query_gate_source_and_receipt_are_frozen():
    files = {str(p) for p in plan.code_files()}
    assert str(Path(query_report.__file__)) in files
    assert str(Path(query_results.__file__)) in files
    gate = plan.accuracy_gate()
    assert gate["review"]["sha256"] == "03ee5dbbdeea3bdfe25636cd9d508292b77f82dc9d302bb77404c6734d03f081"
    assert gate["api"].endswith("query_report.load_cell")

