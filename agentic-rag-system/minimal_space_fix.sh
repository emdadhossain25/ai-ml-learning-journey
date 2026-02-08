#!/bin/bash

echo "=== Minimal Space-Efficient Fix ==="
echo "This will install only what's needed for your existing setup"
echo ""

# Step 1: Downgrade protobuf to work with tensorflow and google-ai
echo "Step 1: Fixing protobuf for tensorflow compatibility..."
pip install --break-system-packages "protobuf==3.20.3"

# Step 2: Install missing packages required by chromadb and gradio
echo ""
echo "Step 2: Installing packages required by chromadb and gradio..."
pip install --break-system-packages fastapi==0.109.0 uvicorn==0.27.0 python-multipart==0.0.9

# Step 3: Install python-dotenv for pydantic-settings
echo ""
echo "Step 3: Installing python-dotenv..."
pip install --break-system-packages python-dotenv==1.0.1

# Step 4: Fix langchain versions to match langchain-community
echo ""
echo "Step 4: Fixing langchain versions..."
pip install --break-system-packages "langchain-core==0.1.52" "langsmith==0.0.87"

# Step 5: Install your requirements
echo ""
echo "Step 5: Installing minimal requirements..."
pip install --break-system-packages -r requirements_minimal.txt

echo ""
echo "=== Verification ==="
pip check

echo ""
echo "=== Installation Complete ==="
