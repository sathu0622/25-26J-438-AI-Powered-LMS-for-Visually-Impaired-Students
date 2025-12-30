#!/bin/bash

echo "Starting Document Processor Backend..."
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install/update dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Start the server
echo ""
echo "Starting FastAPI server..."
echo "Server will be available at http://localhost:8000"
echo ""
python main.py

