#!/bin/bash

echo "=== Dependency Conflict Resolution Script ==="
echo "This script will clean up and reinstall packages with compatible versions"
echo ""

# Step 1: Create a backup of current environment
echo "Step 1: Backing up current pip freeze..."
pip freeze > current_environment_backup.txt

# Step 2: Uninstall conflicting packages
echo ""
echo "Step 2: Uninstalling conflicting packages..."
pip uninstall -y langchain langchain-core langchain-google-genai google-generativeai \
    sentence-transformers faiss-cpu python-dotenv fastapi uvicorn pydantic \
    pypdf python-multipart packaging protobuf

# Step 3: Install updated requirements
echo ""
echo "Step 3: Installing updated requirements..."
pip install --break-system-packages -r requirements_updated.txt

# Step 4: Verify installation
echo ""
echo "Step 4: Verifying installation..."
pip check

echo ""
echo "=== Installation Complete ==="
echo "If you see dependency conflicts above, run: pip check"
echo "Backup of your previous environment saved to: current_environment_backup.txt"
