import ast
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from anygsloc.mapping.build import make_plan, run_plan, contract

ROOT = Path(__file__).resolve().parents[1]


def test_mainline_never_imports_legacy():
    forbidden = {
        "common",
        "data",
        "evidence",
        "features",
        "localization",
        "map_learning",
        "priors",
        "topology",
        "scripts",
        "research",
        "evaluation",
    }
    for path in (ROOT / "anygsloc").rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            modules = []
            if isinstance(node, ast.ImportFrom):
                modules = [node.module or ""]
            if isinstance(node, ast.Import):
                modules = [a.name for a in node.names]
            assert not any(m.split(".")[0] in forbidden for m in modules), path


def test_build_lock_rejects_parallel_resume(tmp_path):
    from anygsloc.mapping.build import exclusive_build

    with exclusive_build(tmp_path):
        with pytest.raises(RuntimeError):
            with exclusive_build(tmp_path):
                pass


def test_build_dry_run_does_not_create_output(tmp_path, capsys):
    from anygsloc.mapping.build import main

    args, prior = build_args(tmp_path)
    main(
        [
            "--dataset",
            args.dataset,
            "--gaussian-ply",
            prior["ply"],
            "--gaussian-type",
            "2dgs",
            "--sh-degree",
            "3",
            "--image-domain",
            "rectified",
            "--output",
            args.output,
            "--dry-run",
        ]
    )
    assert not Path(args.output).exists()
    assert json.loads(capsys.readouterr().out)["schema"] == "anygsloc_build_v1"


def test_extracted_kernel_bodies_preserved():
    manifest = json.loads((ROOT / "research/migration_symbols.json").read_text())
    reverse = {
        record["destination"][:-3].replace("/", "."): module
        for module, record in manifest.items()
    }

    class RestoreImports(ast.NodeTransformer):
        def visit_FunctionDef(self, node):
            # Formatting may clean documentation whitespace; compare executable
            # bodies, not incidental docstring indentation.
            if (
                node.body
                and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Constant)
                and isinstance(node.body[0].value.value, str)
            ):
                node.body = node.body[1:]
            return self.generic_visit(node)

        visit_ClassDef = visit_FunctionDef

        def visit_ImportFrom(self, node):
            node.module = reverse.get(node.module, node.module)
            return node

        def visit_Import(self, node):
            for alias in node.names:
                alias.name = reverse.get(alias.name, alias.name)
            return node

        def visit_Constant(self, node):
            if isinstance(node.value, str):
                node.value = reverse.get(node.value, node.value)
            return node

    count = 0
    for module, record in manifest.items():
        original = ROOT / "research/legacy" / (module.replace(".", "/") + ".py")
        old = {
            n.name: n
            for n in ast.parse(original.read_text()).body
            if isinstance(n, (ast.FunctionDef, ast.ClassDef))
        }
        new = {
            n.name: n
            for n in ast.parse((ROOT / record["destination"]).read_text()).body
            if isinstance(n, (ast.FunctionDef, ast.ClassDef))
        }
        for name in record["symbols"]:
            if module == "data.datasets" and name == "ColmapDataset":
                # Intentional release change: the explicit no-mask evaluator
                # must not deserialize mask files. Check only this exact delta;
                # preserve the AST comparison for every other dataset operation.
                constructor = next(n for n in new[name].body
                                   if isinstance(n, ast.FunctionDef) and n.name == "__init__")
                assert constructor.args.kwonlyargs[-1].arg == "load_masks"
                assert ast.literal_eval(constructor.args.kw_defaults[-1]) is True
                constructor.args.kwonlyargs.pop()
                constructor.args.kw_defaults.pop()
                assignment = constructor.body[-1]
                expected = ast.parse("self._masks = self._load_masks() if load_masks else None").body[0]
                assert ast.dump(assignment) == ast.dump(expected)
                assignment.value = assignment.value.body
            if (
                module == "evidence.parallel_triangulation"
                and name == "_run_fresh_processes"
            ):
                continue
            assert ast.dump(
                RestoreImports().visit(copy.deepcopy(old[name]))
            ) == ast.dump(RestoreImports().visit(copy.deepcopy(new[name]))), (
                module,
                name,
            )
            count += 1
    assert count > 100


def build_args(tmp_path):
    dataset = tmp_path / "dataset"
    (dataset / "sparse/0").mkdir(parents=True)
    (dataset / "sparse/0/cameras.txt").write_text("# fixture metadata\n")
    prior = tmp_path / "prior.ply"
    prior.write_text("ply\n")
    args = SimpleNamespace(
        output=str(tmp_path / "output"),
        dataset=str(dataset),
        images="processed",
        image_domain="rectified",
    )
    return args, dict(ply=str(prior), kind="2dgs", sh_degree=3, white_background=False)


def test_build_is_mapping_only_and_contract_bound(tmp_path):
    args, prior = build_args(tmp_path)
    plan = make_plan(args, prior)
    assert len(plan) == 5
    assert all("evaluate" not in s["command"] for s in plan)
    assert plan[-1]["command"][2] == "anygsloc.cli"
    assert plan[-1]["command"][3] == "compact-map"
    first, digest = contract(args, prior, plan)
    assert not Path(args.output).exists()
    Path(prior["ply"]).write_text("changed input")
    assert contract(args, prior, plan)[1] != digest


def test_resume_requires_hashes_and_skips_finished(tmp_path, monkeypatch):
    artifact = tmp_path / "out/map/result.pt"
    stage = dict(name="map", command=["fixture"], outputs=[str(artifact)])
    calls = []

    def execute(*args, **kwargs):
        assert not artifact.parent.exists()
        artifact.parent.mkdir(parents=True)
        artifact.write_bytes(b"complete")
        calls.append(1)

    monkeypatch.setattr("anygsloc.mapping.build.subprocess.run", execute)
    run_plan([stage], tmp_path / "out", "frozen")
    run_plan([stage], tmp_path / "out", "frozen", resume=True)
    assert len(calls) == 1
    artifact.write_bytes(b"corrupt")
    with pytest.raises(ValueError):
        run_plan([stage], tmp_path / "out", "frozen", resume=True)


def test_interruption_never_counts_as_complete(tmp_path, monkeypatch):
    artifact = tmp_path / "out/result.pt"
    stage = dict(name="map", command=["fixture"], outputs=[str(artifact)])

    def fail(*args, **kwargs):
        raise KeyboardInterrupt()

    monkeypatch.setattr("anygsloc.mapping.build.subprocess.run", fail)
    with pytest.raises(KeyboardInterrupt):
        run_plan([stage], tmp_path / "out", "frozen")
    receipt = json.loads((tmp_path / "out/logs/map.json").read_text())
    assert receipt["status"] == "interrupted"
    artifact.write_bytes(b"partial")
    with pytest.raises(FileExistsError):
        run_plan([stage], tmp_path / "out", "frozen", resume=True)


def test_mapping_calibration_equal_to_legacy():
    from anygsloc.data.calibration import resolve_scene_calibration
    from common.config import resolve_keypoint_count, resolve_reprojection_error_px

    cameras = [
        SimpleNamespace(width=w, height=h, fov_x=1.1, fov_y=0.8)
        for w, h in [(640, 480), (1920, 1080), (1000, 750)]
    ]
    current = resolve_scene_calibration(cameras)
    old = dict(
        keypoints=2048,
        keypoint_reference_area_px=2073600,
        keypoint_minimum=1024,
        keypoint_maximum=2048,
        reprojection_error_px=12.0,
        reprojection_error_angular_tangent=0.007176912978323976,
    )
    assert current["keypoint_count"] == resolve_keypoint_count(old, cameras)
    assert current["reprojection_error_px"] == resolve_reprojection_error_px(
        old, cameras
    )


def test_failure_denominators_are_not_dropped():
    from anygsloc.evaluation.runner import summarize

    rows = [
        dict(
            success=True,
            translation_error_cm=1.0,
            rotation_error_deg=1.0,
            runtime_ms=10.0,
        ),
        dict(
            success=False,
            translation_error_cm=None,
            rotation_error_deg=None,
            runtime_ms=200.0,
        ),
    ]
    summary = summarize(rows, 1.0)
    assert summary["recall_5cm_5deg_percent"] == 50.0
    assert summary["failed_query_count"] == 1
    assert summary["over_100ms_percent"] == 50.0
    assert summary["loop_queries_per_second"] == 2.0
