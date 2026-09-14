"""One-shot, symbol-level repository migration. All edits go through apply_patch.

This is a migration receipt/tool, not a build-time source generator. Run once on
the pre-migration tree; the extracted code is committed as ordinary source.
"""

import ast
import copy
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
LOCAL = {
    "common",
    "data",
    "evidence",
    "features",
    "map_learning",
    "priors",
    "topology",
    "scripts",
    "evaluation",
}
SPECIAL = {
    "common.v6_contracts": "anygsloc.mapping.contracts",
    "common.config": "anygsloc.data.calibration",
    "topology.adaptive_distillation": "anygsloc.mapping.pose_evidence",
    "topology.sufficiency_selector": "anygsloc.mapping.selector",
    "topology.dynamic_reserve": "anygsloc.mapping.observability",
    "topology.matching_coverage": "anygsloc.mapping.coverage",
    "topology.v6_anchor_map": "anygsloc.mapping.anchor_map",
    "map_learning.metric": "anygsloc.mapping.identity",
    "map_learning.v24_anchor_view_support": "anygsloc.mapping.view_support",
    "scripts.materialize_purity_candidates": "anygsloc.mapping.subset",
    "scripts.materialize_v8_v2_projective_map": "anygsloc.mapping.materialize",
    "scripts.probe_rendered_rgb_track_map": "anygsloc.mapping.render_observations",
    "scripts.audit_v7_mapping_render_quality_shard": "anygsloc.mapping.audit_render",
    "evidence.v7_render_certificate": "anygsloc.mapping.render_certificate",
}


def destination(module):
    if module in SPECIAL:
        return SPECIAL[module]
    root, _, rest = module.partition(".")
    area = {
        "common": "utils",
        "data": "data",
        "evidence": "mapping",
        "features": "features",
        "priors": "priors",
        "topology": "mapping",
        "evaluation": "evaluation",
    }
    if root not in area:
        raise ValueError(f"Unexpected research dependency: {module}")
    return "anygsloc." + area[root] + ("." + rest if rest else "")


def local(module):
    return module and module.split(".")[0] in LOCAL


def source_path(module):
    path = ROOT / (module.replace(".", "/") + ".py")
    if not path.exists():
        path = ROOT / module.replace(".", "/") / "__init__.py"
    return path


class Rewrite(ast.NodeTransformer):
    def visit_ImportFrom(self, node):
        if local(node.module):
            node.module = destination(node.module)
        return node

    def visit_Import(self, node):
        for alias in node.names:
            if local(alias.name):
                if not alias.asname:
                    # Existing internal module imports use aliases. Reject a
                    # silent change of binding instead of guessing.
                    raise ValueError(f"Unaliased local import {alias.name}")
                alias.name = destination(alias.name)
        return node

    def visit_Constant(self, node):
        if (
            isinstance(node.value, str)
            and local(node.value)
            and source_path(node.value).is_file()
        ):
            node.value = destination(node.value)
        return node


def rewrite_source(text):
    tree = ast.parse(text)
    edits = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            new = Rewrite().visit(copy.deepcopy(node))
            if ast.dump(new) != ast.dump(node):
                edits.append(
                    (
                        node.lineno - 1,
                        node.col_offset,
                        node.end_lineno - 1,
                        node.end_col_offset,
                        ast.unparse(new),
                    )
                )
        elif (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and local(node.value)
            and source_path(node.value).is_file()
        ):
            edits.append(
                (
                    node.lineno - 1,
                    node.col_offset,
                    node.end_lineno - 1,
                    node.end_col_offset,
                    repr(destination(node.value)),
                )
            )
    lines = text.splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))
    for l, c, el, ec, value in sorted(edits, reverse=True):
        text = text[: offsets[l] + c] + value + text[offsets[el] + ec :]
    return text


def apply_files(files):
    for path, text in files.items():
        target = ROOT / path
        patch = "*** Begin Patch\n"
        if target.exists():
            patch += f"*** Delete File: {path}\n"
        patch += (
            f"*** Add File: {path}\n"
            + "".join("+" + line + "\n" for line in text.splitlines())
            + "*** End Patch\n"
        )
        subprocess.run(
            ["apply_patch"],
            input=patch,
            text=True,
            check=True,
            stdout=subprocess.DEVNULL,
        )


TESTS = [
    "test_camera_input_contract.py",
    "test_projective_association.py",
    "test_projective_completion.py",
    "test_projective_reconstruction.py",
    "test_render_input_contract.py",
    "test_render_valid_superpoint.py",
    "test_sufficiency_selector.py",
    "test_v2_filtered_observations.py",
    "test_virtual_camera_registry.py",
]


def main():
    wanted = {
        "scripts.probe_rendered_rgb_track_map": {"_render_feature_cache"},
        "scripts.audit_v7_mapping_render_quality_shard": {"main"},
        "scripts.materialize_v8_v2_projective_map": {"main"},
        "scripts.materialize_purity_candidates": {"subset"},
        "topology.adaptive_distillation": {
            "_build_pose_evidence",
            "_initial_pose_state",
        },
        "topology.sufficiency_selector": {"HierarchicalSufficiencySelector"},
        "topology.dynamic_reserve": {"spatial_voxel_ids"},
        "data.datasets": {"ColmapDataset"},
        "evaluation.metrics": {"pose_error", "summarize_pose_errors"},
        "features.superpoint": {"SuperPoint"},
        "evidence.parallel_triangulation": {"main"},
    }
    trees = {}
    selected = {}

    def request(module, names):
        wanted.setdefault(module, set()).update(names)

    def imports(nodes):
        for root in nodes:
            for node in ast.walk(root):
                if isinstance(node, ast.ImportFrom) and local(node.module):
                    request(node.module, {x.name for x in node.names})
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if local(alias.name):
                            attrs = {
                                x.attr
                                for x in ast.walk(root)
                                if isinstance(x, ast.Attribute)
                                and isinstance(x.value, ast.Name)
                                and x.value.id == alias.asname
                            }
                            request(alias.name, attrs - {"__file__"})

    test_text = {name: (ROOT / "tests" / name).read_text() for name in TESTS}
    for text in test_text.values():
        imports(ast.parse(text).body)
    previous = None
    while previous != repr(sorted((k, sorted(v)) for k, v in wanted.items())):
        previous = repr(sorted((k, sorted(v)) for k, v in wanted.items()))
        for module, names in list(wanted.items()):
            path = source_path(module)
            text = path.read_text()
            tree = ast.parse(text)
            trees[module] = (text, tree)
            definitions = {}
            for node in tree.body:
                if isinstance(
                    node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
                ):
                    definitions[node.name] = node
                elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                    targets = (
                        node.targets if isinstance(node, ast.Assign) else [node.target]
                    )
                    for target in targets:
                        for n in ast.walk(target):
                            if isinstance(n, ast.Name):
                                definitions[n.id] = node
                elif isinstance(node, (ast.ImportFrom, ast.Import)):
                    for alias in node.names:
                        definitions[alias.asname or alias.name.split(".")[0]] = node
            pending = list(names)
            keep = set()
            while pending:
                name = pending.pop()
                node = definitions.get(name)
                if node is None or node in keep:
                    continue
                keep.add(node)
                pending.extend(
                    n.id
                    for n in ast.walk(node)
                    if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
                )
            # Trim unused names from grouped imports to avoid pulling research
            # branches into the frozen symbol closure.
            used = set(names)
            for node in keep:
                if not isinstance(node, (ast.Import, ast.ImportFrom)):
                    used.update(n.id for n in ast.walk(node) if isinstance(n, ast.Name))
            ordered = []
            for node in tree.body:
                if node not in keep:
                    continue
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    node = copy.deepcopy(node)
                    node.names = [
                        a
                        for a in node.names
                        if (a.asname or a.name.split(".")[0]) in used
                    ]
                    if not node.names:
                        continue
                ordered.append(node)
            selected[module] = ordered
            imports(ordered)
    files = {}
    manifest = {}
    for module, nodes in selected.items():
        text, tree = trees[module]
        lines = text.splitlines(keepends=True)
        # Preserve copyright/license headers verbatim.
        header = "".join(lines[: tree.body[0].lineno - 1])
        chunks = [
            header,
            f'"""Frozen mainline kernel extracted from {module}."""\n',
            "from __future__ import annotations\n",
        ]
        symbols = []
        for node in nodes:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                if isinstance(node, ast.ImportFrom) and node.module == "__future__":
                    continue
                chunks.append(ast.unparse(Rewrite().visit(copy.deepcopy(node))) + "\n")
            else:
                start = (
                    min(
                        [node.lineno]
                        + [d.lineno for d in getattr(node, "decorator_list", [])]
                    )
                    - 1
                )
                chunks.append(rewrite_source("".join(lines[start : node.end_lineno])))
                if hasattr(node, "name"):
                    symbols.append(node.name)
        if "main" in wanted[module]:
            chunks.append("if __name__ == '__main__':\n    main()\n")
        dest = destination(module).replace(".", "/") + ".py"
        files[dest] = "\n\n".join(chunks) + "\n"
        manifest[module] = dict(
            destination=dest,
            source_sha256=hashlib.sha256(text.encode()).hexdigest(),
            symbols=symbols,
        )
    for dest in list(files):
        parent = Path(dest).parent
        while parent != Path("."):
            init = str(parent / "__init__.py")
            if not (ROOT / init).exists():
                files.setdefault(init, '"""AnyGSLoc frozen mainline."""\n')
            parent = parent.parent
    # Leave extracted tests in a staging folder until legacy tests are moved.
    for name, text in test_text.items():
        files["tests_mainline_staging/" + name] = rewrite_source(text)
    files["research/migration_symbols.json"] = json.dumps(manifest, indent=2) + "\n"
    apply_files(files)
    print(
        json.dumps(
            {
                "modules": len(selected),
                "files": len(files),
                "modules_selected": sorted(selected),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
