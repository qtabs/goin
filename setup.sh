#!/bin/bash
# Setup script for goin repository

set -e  # Exit on error

echo "Setting up goin repository..."

# Clone COIN_Python dependency to parent directory if it doesn't exist
if [ ! -d "../COIN_Python" ]; then
    echo "Cloning COIN_Python repository..."
    cd ..
    git clone https://github.com/qtabs/COIN_Python.git
    cd goin
else
    echo "COIN_Python already exists at ../COIN_Python"
fi

# Install Python dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

echo ""
echo "Setup complete!"
echo "You can now run the tests with: python test_goin.py"
