#!/bin/bash

echo "=== Complete Dependency Fix ==="
echo ""

# Install all missing and updated packages
echo "Step 1: Installing langchain-core and protobuf..."
pip install --break-system-packages "langchain-core>=0.3.78,<1.0.0" "protobuf>=5.0,<7.0"

echo ""
echo "Step 2: Installing packaging..."
pip install --break-system-packages "packaging>=24.0"

echo ""
echo "Step 3: Installing all requirements from updated file..."
pip install --break-system-packages -r requirements_updated.txt

echo ""
echo "Step 4: Upgrading pydantic for pydantic-settings..."
pip install --break-system-packages --upgrade "pydantic>=2.7.0"

echo ""
echo "Step 5: Fixing numpy for TensorFlow..."
pip install --break-system-packages "numpy>=1.22,<1.24"

echo ""
echo "=== Final Verification ==="
pip check

echo ""
echo "=== Summary ==="
echo "Installed packages:"
pip list | grep -E "(langchain|protobuf|pydantic|numpy|packaging|multipart)"

echo ""
echo "Done! If you still see conflicts, please share the output above."
