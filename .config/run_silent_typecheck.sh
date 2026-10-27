#!/usr/bin/env bash
set -euo pipefail

# Usage: run_silent_typecheck.sh [file_path] [--verbose|-v]
# Runs ty against .ty/baseline.json, on all files or on file_path only.
# Prints the summary and the first failure, or every failure with --verbose/-v.

file_path=""
verbose=false

for arg in "$@"; do
    case $arg in
        --verbose|-v)
            verbose=true
            ;;
        *)
            if [[ -z "$file_path" ]]; then
                file_path="$arg"
            fi
            ;;
    esac
done

tmp_file=$(mktemp)
trap 'rm -f "$tmp_file"' EXIT

if python "$PIXI_PROJECT_ROOT/.github/scripts/ty_baseline.py" ${file_path:+"$file_path"} > "$tmp_file" 2>&1; then
    printf "  ✓ type checking\n"
    exit 0
fi

printf "  ✗ type checking\n"
if $verbose; then
    cat "$tmp_file"
else
    grep -E '^ty [0-9.]+: ' "$tmp_file" || true
    head -n 1 "$tmp_file"
fi
exit 1
