#!/usr/bin/env python3
import argparse
import sys

from la_artifacts.rgb_teacher_health import check_wildgaussians_checkpoint


def main():
    parser = argparse.ArgumentParser(description="Check RGB teacher checkpoint tensor health.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--backend", default="wildgaussians", choices=["wildgaussians"])
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    health = check_wildgaussians_checkpoint(args.checkpoint)
    if args.json:
        print(health.to_json())
    else:
        print(f"checkpoint={health.checkpoint}")
        print(f"state_path={health.state_path}")
        print(f"ok={health.ok}")
        print(f"reason={health.reason}")
    return 0 if health.ok else 1


if __name__ == "__main__":
    sys.exit(main())
