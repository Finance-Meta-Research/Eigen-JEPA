#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../paper"
pdflatex -interaction=nonstopmode main.tex >/dev/null
pdflatex -interaction=nonstopmode main.tex >/dev/null
