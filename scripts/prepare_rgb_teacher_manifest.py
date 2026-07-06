#!/usr/bin/env python3
import argparse
import os

from la_artifacts.rgb_teacher import (
    RgbTeacherManifest,
    RgbTeacherSpec,
    normalize_render_resolution,
    wildgaussians_render_command_template,
    wildgaussians_train_command,
)


WILDGAUSSIANS_PRESETS = {
    "cambridge_stable_v1": {
        "train_steps": 15000,
        "save_iters": "7000,15000",
        "eval_few_iters": "7000,15000",
        "eval_all_iters": "15000",
        "sets": [
            "appearance_enabled=true",
            "uncertainty_mode=disabled",
            "num_sky_gaussians=50000",
            "densify_until_iter=7000",
        ],
    },
    "cambridge_legacy_noapp_nounc_v1": {
        "train_steps": 15000,
        "save_iters": "7000,15000",
        "eval_few_iters": "7000,15000",
        "eval_all_iters": "15000",
        "sets": [
            "appearance_enabled=false",
            "uncertainty_mode=disabled",
            "num_sky_gaussians=50000",
            "densify_until_iter=7000",
        ],
    },
    "cambridge_app_nounc_v1": {
        "train_steps": 15000,
        "save_iters": "7000,15000",
        "eval_few_iters": "7000,15000",
        "eval_all_iters": "15000",
        "sets": [
            "appearance_enabled=true",
            "uncertainty_mode=disabled",
            "num_sky_gaussians=50000",
            "densify_until_iter=7000",
        ],
    },
    "cambridge_app_dino_v1": {
        "train_steps": 15000,
        "save_iters": "7000,15000",
        "eval_few_iters": "7000,15000",
        "eval_all_iters": "15000",
        "sets": [
            "appearance_enabled=true",
            "uncertainty_mode=dino",
            "num_sky_gaussians=50000",
            "densify_until_iter=7000",
        ],
    },
    "oldhospital_noapp_nosky_30k_v1": {
        "train_steps": 30000,
        "save_iters": "15000,30000",
        "eval_few_iters": "15000,30000",
        "eval_all_iters": "30000",
        "sets": [
            "appearance_enabled=false",
            "uncertainty_mode=disabled",
            "densify_until_iter=7000",
        ],
    },
}


def main():
    parser = argparse.ArgumentParser(description="Create an RGB teacher map manifest for LA-STDLoc.")
    parser.add_argument("--scene", required=True)
    parser.add_argument("--source_path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--output_root", default="/mnt/pool/sqy/stdloc_la_rgb_teacher")
    parser.add_argument("--train_output_path", default="")
    parser.add_argument("--backend", default="wildgaussians", choices=["wildgaussians", "mip-splatting", "inrepo"])
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--fallback_backend", default="mip-splatting")
    parser.add_argument("--nerfbaselines_bin", default=os.environ.get("NERFBASELINES_BIN", "nerfbaselines"))
    parser.add_argument("--nerfbaselines_backend", default=os.environ.get("NERFBASELINES_BACKEND", "conda"))
    parser.add_argument("--train_steps", type=int, default=int(os.environ.get("RGB_TEACHER_TRAIN_STEPS", "0")))
    parser.add_argument("--logger", default=os.environ.get("RGB_TEACHER_LOGGER", "tensorboard"))
    parser.add_argument("--save_iters", default=os.environ.get("RGB_TEACHER_SAVE_ITERS"))
    parser.add_argument("--eval_few_iters", default=os.environ.get("RGB_TEACHER_EVAL_FEW_ITERS"))
    parser.add_argument("--eval_all_iters", default=os.environ.get("RGB_TEACHER_EVAL_ALL_ITERS"))
    parser.add_argument("--render_output_names", default=os.environ.get("RGB_TEACHER_RENDER_OUTPUT_NAMES", "color"))
    parser.add_argument("--render_resolution", default=os.environ.get("WILDGAUSSIANS_RENDER_RESOLUTION", ""))
    parser.add_argument(
        "--wildgaussians_preset",
        default=os.environ.get("RGB_TEACHER_WILDGAUSSIANS_PRESET", ""),
        choices=[""] + sorted(WILDGAUSSIANS_PRESETS),
    )
    parser.add_argument(
        "--wildgaussians_set",
        action="append",
        default=[],
        help="Additional WildGaussians --set KEY=VALUE override. Can be repeated.",
    )
    parser.add_argument("--disable_output_artifact", action="store_true")
    args = parser.parse_args()
    render_resolution = normalize_render_resolution(args.render_resolution)
    train_steps = int(args.train_steps)
    save_iters = args.save_iters
    eval_few_iters = args.eval_few_iters
    eval_all_iters = args.eval_all_iters
    wildgaussians_sets = []
    if args.wildgaussians_preset:
        preset = WILDGAUSSIANS_PRESETS[args.wildgaussians_preset]
        if train_steps <= 0:
            train_steps = int(preset["train_steps"])
        save_iters = save_iters if save_iters is not None else preset["save_iters"]
        eval_few_iters = eval_few_iters if eval_few_iters is not None else preset["eval_few_iters"]
        eval_all_iters = eval_all_iters if eval_all_iters is not None else preset["eval_all_iters"]
        wildgaussians_sets.extend(preset["sets"])
    wildgaussians_sets.extend(args.wildgaussians_set)

    train_command = []
    render_command = []
    if args.backend == "wildgaussians":
        train_command = wildgaussians_train_command(
            args.source_path,
            args.output_root,
            args.scene,
            output_path=args.train_output_path,
            nerfbaselines_bin=args.nerfbaselines_bin,
            nerfbaselines_backend=args.nerfbaselines_backend,
            train_steps=train_steps,
            logger=args.logger,
            save_iters=save_iters,
            eval_few_iters=eval_few_iters,
            eval_all_iters=eval_all_iters,
            config_sets=wildgaussians_sets,
            disable_output_artifact=args.disable_output_artifact,
        )
        render_command = wildgaussians_render_command_template(
            args.checkpoint,
            nerfbaselines_bin=args.nerfbaselines_bin,
            nerfbaselines_backend=args.nerfbaselines_backend,
            output_names=args.render_output_names,
            resolution=render_resolution,
        )

    spec = RgbTeacherSpec(
        scene=args.scene,
        source_path=os.path.abspath(args.source_path),
        backend=args.backend,
        checkpoint=os.path.abspath(args.checkpoint) if args.checkpoint else "",
        output_root=os.path.abspath(args.output_root),
        nerfbaselines_bin=args.nerfbaselines_bin,
        nerfbaselines_backend=args.nerfbaselines_backend,
        train_command=train_command,
        render_command_template=render_command,
        fallback_backend=args.fallback_backend,
        status="ready" if args.checkpoint else "planned",
    )
    ok, reason = spec.validate()
    spec.metrics.update({"validation_ok": ok, "validation_reason": reason})
    if args.wildgaussians_preset:
        spec.metrics["wildgaussians_preset"] = args.wildgaussians_preset
    if wildgaussians_sets:
        spec.metrics["wildgaussians_sets"] = wildgaussians_sets
    if not ok and spec.status == "ready":
        spec.status = "invalid"
    manifest = RgbTeacherManifest.single(spec)
    manifest.save(args.output)
    print(f"Wrote RGB teacher manifest: {args.output}")
    print(f"backend={args.backend} validation_ok={ok} reason={reason}")
    if train_command:
        print("train_command:", " ".join(train_command))


if __name__ == "__main__":
    main()
