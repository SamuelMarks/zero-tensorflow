#!/bin/bash
set -euo pipefail
export UV_DYNAMIC_VERSIONING_BYPASS="2.2.0"
export PDM_BUILD_SCM_VERSION="2.2.0"
echo "Setting up local editable environment..."
pip install "ml-ecosystem-snapshots @ https://github.com/SamuelMarks/ml-ecosystem-snapshots/releases/download/v0.0.5/ml_ecosystem_snapshots-0.0.5-py3-none-any.whl"
mkdir -p ~/.cache/ml_ecosystem_snapshots
curl -sL "https://github.com/SamuelMarks/ml-ecosystem-snapshots/releases/download/v0.0.5/tensorflow_v2.21.0.json" -o ~/.cache/ml_ecosystem_snapshots/tensorflow_v2.21.0.json
pip install -e ../ml-switcheroo-ir
pip install -e ../ml-switcheroo-compiler
pip install -e ../zero-keras
pip install -e .
echo "Done!"
