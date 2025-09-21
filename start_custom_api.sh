#!/bin/bash

echo "🚀 Starting Vector Store RAG - Board Game Recommendations API"
echo "=============================================================="

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 is not installed or not in PATH"
    exit 1
fi

# Install requirements if needed
echo "📦 Installing custom API requirements..."
pip3 install -r requirements_custom_api.txt

# Start Langflow in the background
echo "🔧 Starting Langflow..."
langflow run --host 0.0.0.0 --port 7861 &
LANGFLOW_PID=$!

# Wait for Langflow to start
echo "⏳ Waiting for Langflow to start..."
sleep 10

# Check if Langflow is running
if ! kill -0 $LANGFLOW_PID 2>/dev/null; then
    echo "❌ Failed to start Langflow"
    exit 1
fi

echo "✅ Langflow started successfully (PID: $LANGFLOW_PID)"

# Start the custom API server
echo "🌐 Starting custom API server..."
python3 custom_api_server.py &
API_PID=$!

# Wait for API server to start
sleep 3

echo ""
echo "🎉 Vector Store RAG API is now running!"
echo "========================================"
echo "📍 Custom API Server: http://localhost:5000"
echo "📍 Langflow Backend: http://localhost:7861"
echo "🎯 Target Flow: ${TARGET_FLOW_NAME:-Better Board Game Search} (ID: ${TARGET_FLOW_ID:-your-flow-id-here})"
echo ""
echo "🔗 API Endpoints:"
echo "   • Get flows: http://localhost:5000/api/v1/flows/"
echo "   • Run flow: http://localhost:5000/api/v1/run/${TARGET_FLOW_ID:-your-flow-id-here}"
echo "   • Health check: http://localhost:5000/health"
echo ""
echo "🔧 To stop the services, press Ctrl+C"
echo ""

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "🛑 Stopping services..."
    kill $API_PID 2>/dev/null
    kill $LANGFLOW_PID 2>/dev/null
    echo "✅ Services stopped"
    exit 0
}

# Set trap to cleanup on script exit
trap cleanup SIGINT SIGTERM

# Wait for processes
wait
