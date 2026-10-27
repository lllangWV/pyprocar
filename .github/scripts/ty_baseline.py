"""Fail on a ty diagnostic the baseline lacks, and on a baseline entry ty no longer reports.

Usage: python .github/scripts/ty_baseline.py [--prune | --regenerate] [PATH ...]

ty has no baseline of its own. .ty/baseline.json lists, per file, the rule and message
of each diagnostic the code had when ty replaced basedpyright. Entries carry no line
numbers, so an edit that only moves a diagnostic does not fail.

PATH limits the check to those files and directories, and to their baseline entries.
--prune deletes the stale entries and adds none.
--regenerate rewrites the whole baseline from ty's output. Use it only in the commit
             that changes the ty version in pixi.toml, because each ty release
             rewords and adds diagnostics.
"""

import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

BASELINE = Path(".ty/baseline.json")

Diagnostic = tuple[str, str]  # (rule, message)


def ty_version() -> str:
    result = subprocess.run(["ty", "--version"], capture_output=True, text=True, check=True)
    return result.stdout.split()[1]


def run_ty(paths: list[str]) -> list[dict]:
    cmd = ["ty", "check", "--output-format", "gitlab", "--exit-zero", *paths]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode:
        sys.exit(f"ty failed (exit {result.returncode}):\n{result.stderr}")
    return json.loads(result.stdout)


def key(report: dict) -> Diagnostic:
    rule = report["check_name"]
    return rule, report["description"].removeprefix(f"{rule}: ")


def by_file(reports: list[dict]) -> dict[str, Counter[Diagnostic]]:
    found: dict[str, Counter[Diagnostic]] = {}
    for report in reports:
        found.setdefault(report["location"]["path"], Counter())[key(report)] += 1
    return found


def compare(
    found: dict[str, Counter[Diagnostic]], baseline: dict[str, Counter[Diagnostic]]
) -> tuple[dict[str, Counter[Diagnostic]], dict[str, Counter[Diagnostic]]]:
    """Return (new, stale): diagnostics beyond the baseline, and entries ty no longer reports."""
    new = {path: found[path] - baseline.get(path, Counter()) for path in found}
    stale = {path: baseline[path] - found.get(path, Counter()) for path in baseline}
    return {p: c for p, c in new.items() if c}, {p: c for p, c in stale.items() if c}


def load(path: Path) -> tuple[str, dict[str, Counter[Diagnostic]]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    files = {
        f: Counter((e["rule"], e["message"]) for e in entries)
        for f, entries in data["files"].items()
    }
    return data["ty"], files


def dump(path: Path, version: str, files: dict[str, Counter[Diagnostic]]) -> None:
    entries = {
        f: [{"rule": rule, "message": message} for rule, message in sorted(files[f].elements())]
        for f in sorted(files)
        if files[f]
    }
    path.parent.mkdir(exist_ok=True)
    path.write_text(
        json.dumps({"ty": version, "files": entries}, indent=1) + "\n", encoding="utf-8"
    )


def under(path: str, roots: list[str]) -> bool:
    return not roots or any(
        Path(path).resolve().is_relative_to(Path(root).resolve()) for root in roots
    )


def main(args: list[str]) -> int:
    mode = args.pop(0) if args and args[0] in ("--prune", "--regenerate") else None
    version = ty_version()
    reports = run_ty(args)
    found = by_file(reports)
    if mode == "--regenerate":
        if args:
            sys.exit("--regenerate rewrites the whole baseline and takes no PATH")
        dump(BASELINE, version, found)
        print(f"{BASELINE}: {len(reports)} diagnostics from ty {version}")
        return 0

    recorded, baseline = load(BASELINE)
    if recorded != version:
        sys.exit(
            f"{BASELINE} was written by ty {recorded}, the env has ty {version}. "
            "Pin ty to the baseline's version, "
            "or regenerate the baseline in the commit that changes the pin."
        )
    in_scope = {f: c for f, c in baseline.items() if under(f, args)}
    new, stale = compare(found, in_scope)

    if mode == "--prune":
        for f, gone in stale.items():
            baseline[f] -= gone
        dump(BASELINE, version, baseline)
        print(f"{BASELINE}: deleted {sum(c.total() for c in stale.values())} stale entries")
        return 0

    for report in reports:
        path = report["location"]["path"]
        if new.get(path, Counter())[key(report)]:
            begin = report["location"]["positions"]["begin"]
            print(f"{path}:{begin['line']}:{begin['column']}: {report['description']}")
    for f, gone in stale.items():
        for rule, message in gone.elements():
            print(f"{f}: stale baseline entry, ty no longer reports {rule}: {message}")
    n_new = sum(c.total() for c in new.values())
    n_stale = sum(c.total() for c in stale.values())
    print(
        f"ty {version}: {len(reports)} diagnostics, {n_new} new, {n_stale} stale baseline entries"
    )
    if n_new:
        print("Fix each new diagnostic. Never add it to the baseline.")
    if n_stale:
        print(f"Delete the stale entries with: python {sys.argv[0]} --prune")
    return int(bool(n_new or n_stale))


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
