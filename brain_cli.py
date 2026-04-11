from __future__ import annotations

import argparse
import json
from typing import Optional

from brain_roundtrip import BrainPipeline, print_result
from brain_commands import BrainCommands


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="brain_cli",
        description="BrAIn command-line interface",
    )

    subparsers = parser.add_subparsers(dest="mode", required=True)

    ingest_parser = subparsers.add_parser("ingest", help="Ingest text into BrAIn")
    ingest_parser.add_argument("text", type=str, help="Text to ingest")
    ingest_parser.add_argument("--actor", type=str, default="user", help="Actor label for the input")
    ingest_parser.add_argument("--save", action="store_true", help="Save output artifacts to brain_output")
    ingest_parser.add_argument("--export-images", action="store_true", help="Export PNG versions of the core and relation grids")
    ingest_parser.add_argument("--base-name", type=str, default="brain_cli_run", help="Base file name for saved artifacts")

    command_parser = subparsers.add_parser("command", help="Run a BrAIn maintenance command")
    command_parser.add_argument("command_text", type=str, help="Command text")

    search_parser = subparsers.add_parser("search", help="Search BrAIn history and artifacts")
    search_parser.add_argument("query", type=str, help="Search query")

    subparsers.add_parser("show-pending", help="Show pending merge suggestions")
    subparsers.add_parser("show-approved", help="Show approved merge suggestions")
    subparsers.add_parser("show-rejected", help="Show rejected merge suggestions")
    subparsers.add_parser("show-history", help="Show recent history")
    subparsers.add_parser("clear-history", help="Clear history")
    subparsers.add_parser("show-artifacts", help="Show recent artifacts")

    approve_parser = subparsers.add_parser("approve", help="Approve a merge suggestion")
    approve_parser.add_argument("left", type=str)
    approve_parser.add_argument("right", type=str)

    reject_parser = subparsers.add_parser("reject", help="Reject a merge suggestion")
    reject_parser.add_argument("left", type=str)
    reject_parser.add_argument("right", type=str)

    apply_parser = subparsers.add_parser("apply-approved", help="Apply approved merges to the alias map")
    apply_parser.add_argument("--prefer", choices=["left", "right"], default="right", help="Choose canonical side")

    return parser


def run_ingest(
    pipeline: BrainPipeline,
    text: str,
    actor: str,
    save: bool,
    export_images: bool,
    base_name: str,
) -> int:
    result = pipeline.run(text, actor=actor)
    print_result(result)

    if save:
        paths = pipeline.save_artifacts(
            result,
            base_name=base_name,
            export_images=export_images,
        )
        print("\nSAVED ARTIFACTS")
        print(json.dumps(paths, indent=2))

    return 0


def run_command(commands: BrainCommands, command_text: str) -> int:
    result = commands.run(command_text)

    print("=" * 72)
    print("COMMAND")
    print(command_text)
    print("\nRESULT")
    print(result.message)
    print(json.dumps(result.data, indent=2))

    return 0 if result.ok else 1


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    pipeline = BrainPipeline()
    commands = BrainCommands(pipeline=pipeline)

    if args.mode == "ingest":
        return run_ingest(
            pipeline=pipeline,
            text=args.text,
            actor=args.actor,
            save=args.save,
            export_images=args.export_images,
            base_name=args.base_name,
        )

    if args.mode == "command":
        return run_command(commands, args.command_text)

    if args.mode == "search":
        return run_command(commands, f"search {args.query}")

    if args.mode == "show-pending":
        return run_command(commands, "show pending merges")

    if args.mode == "show-approved":
        return run_command(commands, "show approved merges")

    if args.mode == "show-rejected":
        return run_command(commands, "show rejected merges")

    if args.mode == "show-history":
        return run_command(commands, "show recent history")

    if args.mode == "clear-history":
        return run_command(commands, "clear history")

    if args.mode == "show-artifacts":
        return run_command(commands, "show recent artifacts")

    if args.mode == "approve":
        return run_command(commands, f"approve merge {args.left} {args.right}")

    if args.mode == "reject":
        return run_command(commands, f"reject merge {args.left} {args.right}")

    if args.mode == "apply-approved":
        result = pipeline.apply_approved_merges(prefer=args.prefer)
        print("=" * 72)
        print("APPLY APPROVED MERGES")
        print(json.dumps(result, indent=2))
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())