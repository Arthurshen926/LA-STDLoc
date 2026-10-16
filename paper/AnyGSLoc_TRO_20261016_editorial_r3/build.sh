#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
mkdir -p checks
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex > checks/main_build.txt 2>&1
printf 'Built %s/main.pdf\n' "$PWD"
