"""CLI integration for BeforeCommit replay generation."""

from __future__ import annotations

import argparse
import json

from zsc_identifiability.demo.exporter import capture_collection, export_site_data
from zsc_identifiability.demo.validator import validate_site_data


def add_demo_parser(commands: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    demo = commands.add_parser("demo", help="capture, export, or validate BeforeCommit replays")
    subcommands = demo.add_subparsers(dest="demo_command", required=True)
    capture = subcommands.add_parser("capture", help="capture the curated frozen episodes")
    capture.add_argument("--curation", required=True)
    capture.add_argument("--output", required=True)
    export = subcommands.add_parser("export-site", help="export replay packages for the web app")
    export.add_argument("--replays", required=True)
    export.add_argument("--output", required=True)
    validate = subcommands.add_parser("validate", help="validate deployable replay data")
    validate.add_argument("--site-data", required=True)


def dispatch_demo(args: argparse.Namespace) -> int:
    if args.demo_command == "capture":
        files = capture_collection(args.curation, args.output)
        payload = {"status": "complete", "replay_packages": [str(path) for path in files]}
    elif args.demo_command == "export-site":
        files = export_site_data(args.replays, args.output)
        payload = {"status": "complete", "written_files": [str(path) for path in files]}
    else:
        payload = validate_site_data(args.site_data)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0
