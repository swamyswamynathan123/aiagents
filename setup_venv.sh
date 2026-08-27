#!/bin/bash
set -e
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
echo ""
echo "Virtual environment ready."
echo "Run: source venv/bin/activate"
