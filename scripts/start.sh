#!/bin/bash
echo "🎭 Starting Emotion AI System..."

# Get the project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Check if npm is installed
if ! command -v npm &> /dev/null; then
    echo "❌ Error: npm is not installed. Please install Node.js."
    exit 1
fi

# Build Frontend
echo "🏗️  Building Frontend..."
cd src/web/frontend
npm install
npm run build
if [ $? -ne 0 ]; then
    echo "❌ Frontend build failed."
    exit 1
fi
cd "$PROJECT_ROOT"

# Start Backend
echo "🚀 Starting Web Server..."
python run_web.py --no-auto-start
