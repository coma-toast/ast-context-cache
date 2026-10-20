#!/usr/bin/env bash
# Prints the version the next release gets: the patch after the newest vX.Y.Z tag, or the
# VERSION file when someone raised it past that tag (how a PR asks for a minor or major
# bump). main is protected, so releases are tags only and VERSION is never auto-committed.
set -euo pipefail
file="${1:-VERSION}"
floor=""
[[ -f "$file" ]] && floor=$(tr -d '[:space:]' < "$file")
latest=$(git tag -l 'v[0-9]*.[0-9]*.[0-9]*' --sort=-v:refname | head -n1)
latest=${latest#v}
if [[ -z "$latest" ]]; then
  echo "${floor:-0.0.1}"
  exit 0
fi
IFS=. read -r major minor patch <<< "$latest"
next="${major}.${minor}.$((patch + 1))"
if [[ -n "$floor" && "$(printf '%s\n%s\n' "$floor" "$next" | sort -V | tail -n1)" == "$floor" && "$floor" != "$latest" ]]; then
  next="$floor"
fi
echo "$next"
