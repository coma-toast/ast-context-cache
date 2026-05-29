#!/usr/bin/env bash
set -euo pipefail
file="${1:-VERSION}"
current=$(tr -d '[:space:]' < "$file")
IFS=. read -r major minor patch <<< "$current"
patch=$((patch + 1))
echo "${major}.${minor}.${patch}" > "$file"
tr -d '[:space:]' < "$file"
