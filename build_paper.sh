#!/bin/sh
set -eu
cd "$(dirname "$0")/paper"
mkdir -p build
tectonic --keep-logs --outdir build manuscript.tex
