#!/bin/bash
echo "Setting up local editable environment..."
pip install -e .
pip install -e ../ml-switcheroo-compiler
pip install -e ../zero-keras
echo "Done!"
