"""Run native STDLoc/ULF-Loc with the paper's mask and timing adapters."""

import argparse
import json
import resource
import runpy
from pathlib import Path

from anygsloc.pipeline.benchmark import selection
from reproduction.native_hooks import (
    NativeRecorder,
    mask_rgb,
    std_text,
    transform_native,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["accuracy", "timing", "train"])
    parser.add_argument("--method", choices=["STDLoc", "ULFLoc"], required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument(
        "--model", type=Path, help="Existing native model for accuracy/timing"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mapping-semantics", choices=["on", "off"], default="on")
    parser.add_argument(
        "--scene",
        choices=[
            "ShopFacade",
            "OldHospital",
            "KingsCollege",
            "GreatCourt",
            "StMarysChurch",
        ],
    )
    parser.add_argument("--round", type=int, choices=[0, 1, 2], default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    wrapper = root / "reproduction/native_wrapper.py"
    repo, dataset, output = (
        args.repo.resolve(),
        args.dataset.resolve(),
        args.output.resolve(),
    )
    model = output / "model"
    receipt = output / "receipt.json"
    # ULF-Loc chooses its dataset reader from a case-sensitive path substring.
    native_dataset = (
        output / "inputs/cambridge" / dataset.name
        if args.method == "ULFLoc"
        else dataset
    )
    phase = "train" if args.stage == "train" else "evaluate"
    native_args = [
        "-s",
        str(native_dataset),
        "-m",
        str(model),
        "--images",
        str(native_dataset / "processed"),
        "--data_device",
        "cpu",
        "--cfg",
        str(args.config.resolve()),
        "--longest_edge",
        "640",
    ]
    argv = [
        str(wrapper),
        "--repo",
        str(repo),
        "--source",
        str(native_dataset),
        "--method",
        args.method,
        "--mapping-semantics",
        args.mapping_semantics,
        "--phase",
        phase,
        "--receipt",
        str(receipt),
        "--run-tag",
        "query_semantic_on",
        "--",
        *native_args,
    ]
    if args.dry_run:
        print(
            json.dumps(
                {
                    "argv": argv,
                    "query_semantics": args.stage != "train",
                    "stage": args.stage,
                    "round": args.round,
                },
                indent=2,
            )
        )
        return
    if args.stage != "train" and args.model is None:
        parser.error("--model is required for evaluation")
    if args.stage == "timing" and not args.scene:
        parser.error("--scene is required for the fixed timing subset")
    import torch

    from anygsloc.data.datasets import ColmapDataset
    from anygsloc.artifacts import sha256
    from anygsloc.pipeline.io import write_json

    output.mkdir(parents=True, exist_ok=False)
    if native_dataset != dataset:
        native_dataset.parent.mkdir(parents=True)
        native_dataset.symlink_to(dataset, target_is_directory=True)
    write_json(output / "command.json", {"argv": argv, "stage": args.stage})
    inventory = {}
    if args.stage != "train":
        files = ["cfg_args", "point_cloud/iteration_30000/point_cloud.ply"]
        files += (
            ["detector/sampled_idx.pkl", "detector/30000_detector.pth"]
            if args.method == "STDLoc"
            else ["test/keypoints_sampled_idx.pkl", "test/keypoints_features.pkl"]
        )
        for relative in files:
            source = args.model.resolve() / relative
            if not source.is_file():
                raise FileNotFoundError(source)
            target = model / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.symlink_to(source)
            inventory[relative] = {
                "bytes": source.stat().st_size,
                "sha256": sha256(source),
            }
    names = [
        c.image_name for c in ColmapDataset(dataset, load_masks=False).split("test")
    ]
    native_output = (
        output / "native_output"
        if args.method == "STDLoc"
        else model / "query_semantic_on"
    )
    write_json(
        output / "manifest.json",
        {
            "method": args.method,
            "scene": args.scene or dataset.name,
            "mapping_semantics": args.mapping_semantics,
            "query_names": names,
            "stage": args.stage,
            "model_inventory": inventory,
            "storage_bytes": sum(item["bytes"] for item in inventory.values()),
            "results": str((native_output / "results.json").relative_to(output)),
        },
    )
    recorder = None
    if args.stage == "timing":
        selected = selection(args.scene, names)
        recorder = NativeRecorder(
            selected,
            args.round,
            sync=torch.cuda.synchronize,
            reset_peak=torch.cuda.reset_peak_memory_stats,
            memory=lambda: {
                "allocated_bytes": torch.cuda.max_memory_allocated(),
                "reserved_bytes": torch.cuda.max_memory_reserved(),
            },
        )

    def execute_native(path, method, masks):
        if method == "ULFLoc" and recorder is None:
            return runpy.run_path(path, run_name="__main__")
        scope = {
            "__name__": "__main__",
            "__file__": path,
            "__package__": "",
            "_CONTROLLED_OUTPUT": str(native_output),
            "_MATCHED": recorder,
            "_QUERY_MASK_RGB": lambda name, image: mask_rgb(
                name, image, masks, set(names)
            ),
        }
        text = Path(path).read_text()
        text = (
            transform_native(text, method) if recorder is not None else std_text(text)
        )
        exec(compile(text, path, "exec"), scope)
        return scope

    from reproduction.native_wrapper import main as run_native_wrapper

    run_native_wrapper(argv[1:], native_runner=execute_native, recorder=recorder)
    if recorder is not None:
        result = recorder.result()
        result.update(
            scene=args.scene,
            method=args.method,
            mapping_semantics=args.mapping_semantics,
            selection=selected,
            storage_bytes=sum(item["bytes"] for item in inventory.values()),
            process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            versions={"torch": torch.__version__, "cuda": torch.version.cuda},
            device=torch.cuda.get_device_name(),
            segmentation_inference_included=False,
        )
        write_json(output / "round.json", result)


if __name__ == "__main__":
    main()
