#!/usr/bin/env python3
"""Partition actual pytest collection by whole files for CI; local tests stay unchanged."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pytest


class FileShard:
    def __init__(self, engine_root: Path, index: int, count: int, report: Path | None):
        self.engine_root = engine_root
        self.index = index
        self.count = count
        self.report = report
        self.inventory: dict[str, object] | None = None

    def write_report(self, exit_code: int | None = None) -> None:
        if self.report is None or self.inventory is None:
            return
        if exit_code is not None:
            self.inventory["exit_code"] = exit_code
        self.report.parent.mkdir(parents=True, exist_ok=True)
        self.report.write_text(json.dumps(self.inventory, indent=2) + "\n")

    @pytest.hookimpl(trylast=True)
    def pytest_collection_modifyitems(self, session, config, items) -> None:
        file_for = {
            item.nodeid: Path(item.path).resolve().relative_to(self.engine_root).as_posix()
            for item in items
        }
        files = sorted(set(file_for.values()))
        selected_files = files[self.index :: self.count]
        selected_set = set(selected_files)
        selected = [item for item in items if file_for[item.nodeid] in selected_set]
        deselected = [item for item in items if file_for[item.nodeid] not in selected_set]
        self.inventory = {
            "schema_version": 1,
            "shard_index": self.index,
            "shard_count": self.count,
            "all_files": files,
            "selected_files": selected_files,
            "all_nodeids": [item.nodeid for item in items],
            "selected_nodeids": [item.nodeid for item in selected],
        }
        self.write_report()
        if not selected:
            raise pytest.UsageError(f"Engine shard {self.index}/{self.count} contains no tests.")
        items[:] = selected
        config.hook.pytest_deselected(items=deselected)
        message = (
            f"Engine shard {self.index}/{self.count}: "
            f"{len(selected_files)}/{len(files)} files, "
            f"{len(selected)}/{len(selected) + len(deselected)} cases"
        )
        terminal = config.pluginmanager.get_plugin("terminalreporter")
        if terminal is not None:
            terminal.write_line(message)
        else:
            print(message, flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=int, required=True, help="Zero-based shard index.")
    parser.add_argument("--count", type=int, required=True, help="Total number of shards.")
    parser.add_argument("--collect-only", action="store_true", help="Report coverage without executing tests.")
    parser.add_argument("--report", type=Path, help="Write the selected and complete node inventories as JSON.")
    parser.add_argument(
        "--engine-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "components/engine",
        help="Engine project root; defaults to this checkout's Engine.",
    )
    args = parser.parse_args()
    if args.count < 1 or not 0 <= args.index < args.count:
        parser.error("--count must be positive and --index must be within [0, count).")
    engine_root = args.engine_root.resolve()
    if not (engine_root / "pyproject.toml").is_file() or not (engine_root / "tests").is_dir():
        parser.error("--engine-root must contain the Engine pyproject.toml and tests directory.")
    report = args.report.resolve() if args.report else None
    os.chdir(engine_root)
    shard = FileShard(engine_root, args.index, args.count, report)
    options = ["--collect-only"] if args.collect_only else []
    result = int(pytest.main(options, plugins=[shard]))
    shard.write_report(result)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
