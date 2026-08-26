"""Legacy asset and command audit without importing it into the main package."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    parser.add_argument("--result", required=True)
    args = parser.parse_args()
    request_path = Path(args.request).resolve()
    result_path = Path(args.result).resolve()
    request = json.loads(request_path.read_text())
    expected = request.pop("request_hash")
    observed = hashlib.sha256(
        json.dumps(request, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if observed != expected:
        raise ValueError("legacy runtime request hash mismatch")
    root = _project_root(request_path)
    _verify_upstreams(request, root)
    operation = request["operation"]
    if operation == "validate":
        payload = {"imports_deferred": True, "pins_verified": True}
    elif operation == "audit_assets":
        payload = _audit_assets(request["payload"], root)
    elif operation == "tomzsc_command":
        payload = _tomzsc_command(request["payload"], root, request)
    else:
        raise ValueError(f"unsupported legacy runtime operation: {operation!r}")
    result = {
        "schema_version": 1,
        "request_hash": expected,
        "operation": operation,
        "status": payload.get("status", "complete"),
        "python_version": platform.python_version(),
        "payload": payload,
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


def _audit_assets(payload, root):
    pool = (root / payload["policy_pool_path"]).resolve()
    if not pool.is_dir():
        return {
            "status": "secondary_unavailable",
            "reason": "official policy pool directory is absent",
            "partner_count": 0,
            "algorithm_count": 0,
        }
    index_path = pool / "stage6-assets.json"
    if index_path.is_file():
        index = json.loads(index_path.read_text())
        partner_count = len(index.get("evaluation_partners", []))
        algorithm_count = len(index.get("algorithms", []))
        response_assets = bool(index.get("best_response_assets"))
        discovery_mode = "versioned_index"
        source_hash = hashlib.sha256(index_path.read_bytes()).hexdigest()
    else:
        discovered = _discover_official_assets(pool)
        partner_count = discovered["partner_count"]
        algorithm_count = discovered["algorithm_count"]
        response_assets = discovered["best_response_assets"]
        discovery_mode = "official_tree_scan"
        source_hash = discovered["tree_hash"]
    available = (
        partner_count >= int(payload["minimum_partners"])
        and algorithm_count >= int(payload["minimum_algorithms"])
        and response_assets
    )
    return {
        "status": "complete" if available else "secondary_unavailable",
        "partner_count": partner_count,
        "algorithm_count": algorithm_count,
        "best_response_assets": response_assets,
        "discovery_mode": discovery_mode,
        "asset_hash": source_hash,
    }


def _discover_official_assets(pool):
    policy_files = tuple(
        sorted(
            path
            for path in pool.rglob("*")
            if path.is_file() and path.suffix.lower() in {".pt", ".pth", ".pkl"}
        )
    )
    relative_tokens = {
        path: set(
            token
            for token in re.split(r"[^a-z0-9]+", str(path.relative_to(pool)).lower())
            if token
        )
        for path in policy_files
    }
    algorithm_names = {
        name
        for name in (
            "sp",
            "fcp",
            "mep",
            "trajedi",
            "cole",
            "e3t",
            "hsp",
            "other_play",
        )
        if any(name in relative_tokens[path] for path in policy_files)
    }
    partner_files = tuple(
        path
        for path in policy_files
        if any(
            marker in str(path.relative_to(pool)).lower()
            for marker in ("bias", "partner", "eval", "population")
        )
    )
    response_files = tuple(
        path
        for path in policy_files
        if any(
            marker in str(path.relative_to(pool)).lower()
            for marker in ("best_response", "best-response", "_br", "/br")
        )
    )
    digest = hashlib.sha256()
    for path in policy_files:
        digest.update(str(path.relative_to(pool)).encode())
        digest.update(str(path.stat().st_size).encode())
    partner_units = {str(path.parent.relative_to(pool)) for path in partner_files}
    return {
        "partner_count": len(partner_units),
        "algorithm_count": len(algorithm_names),
        "best_response_assets": bool(response_files),
        "tree_hash": digest.hexdigest(),
    }


def _tomzsc_command(payload, root, request):
    repository = request["upstreams"]["tomzsc"]
    workdir = (root / repository["path"]).resolve()
    allowed = {
        "cross_play": "overcooked_cross_play.py",
        "cluster": "clustering/get_clusters.py",
        "train_tom": "overcooked_train_tom.py",
        "evaluate": "overcooked_eval.py",
    }
    stage = payload["stage"]
    if stage not in allowed:
        raise ValueError(f"unsupported ToMZSC stage: {stage!r}")
    command = ["python", allowed[stage], *[str(value) for value in payload.get("arguments", [])]]
    completed = subprocess.run(command, cwd=workdir, text=True, capture_output=True)
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or completed.stdout.strip())
    return {"stage": stage, "status": "complete", "command": command}


def _verify_upstreams(request, root):
    for repository in request["upstreams"].values():
        path = (root / repository["path"]).resolve()
        observed = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
            capture_output=True,
            check=True,
        ).stdout.strip()
        if observed != repository["commit"]:
            raise ValueError(f"upstream pin mismatch at {path}")


def _project_root(request_path):
    for parent in request_path.parents:
        if (parent / "pyproject.toml").is_file() and (parent / "src/zsc_identifiability").is_dir():
            return parent
    raise ValueError("could not locate zsc-identifiability project root")


if __name__ == "__main__":
    main()
