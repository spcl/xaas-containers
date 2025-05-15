#!/bin/bash

set -e  # Exit on error

echo "Setting up Python environment..."

# Ensure pip is installed and up-to-date
python3 -m ensurepip --upgrade || true
python3 -m pip install --upgrade pip

# Install required packages
pip install -U google-generativeai==0.8.3
pip install packaging

# Configure Gemini API key (if not set externally)
if [[ -z "$GOOGLE_API_KEY" ]]; then
    echo "WARNING: GOOGLE_API_KEY is not set. Please edit this script or export it in your environment."
    export GOOGLE_API_KEY="ADD-KEY-HERE"
fi

echo "Environment setup complete!"