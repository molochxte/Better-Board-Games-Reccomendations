#!/usr/bin/env python3
"""
Custom API Server for Vector Store RAG - Board Game Recommendations
This server acts as a proxy to Langflow, only exposing the specific Vector Store RAG flow.
"""

import os
import json
import requests
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configuration
LANGFLOW_BASE_URL = os.getenv("LANGFLOW_BASE_URL", "http://localhost:7861")
TARGET_FLOW_ID = os.getenv("TARGET_FLOW_ID", "your-flow-id-here")  # Your Better Board Game Search flow
TARGET_FLOW_NAME = os.getenv("TARGET_FLOW_NAME", "Better Board Game Search")

# Headers to pass through to Langflow
LANGFLOW_HEADERS = {
    'Content-Type': 'application/json',
    'Accept': 'application/json'
}

@app.route('/api/v1/version', methods=['GET'])
def get_version():
    """Get Langflow version information."""
    try:
        response = requests.get(f"{LANGFLOW_BASE_URL}/api/v1/version", headers=LANGFLOW_HEADERS)
        return jsonify(response.json()), response.status_code
    except Exception as e:
        logger.error(f"Error getting version: {e}")
        return jsonify({"error": "Failed to get version"}), 500

@app.route('/api/v1/flows/', methods=['GET'])
def get_flows():
    """Return only the Vector Store RAG flow."""
    try:
        # Get all flows from Langflow
        response = requests.get(f"{LANGFLOW_BASE_URL}/api/v1/flows/", headers=LANGFLOW_HEADERS)
        
        if response.status_code == 200:
            flows = response.json()
            
            # Filter to only include the Better Board Game Search flow
            filtered_flows = [
                flow for flow in flows 
                if flow.get('id') == TARGET_FLOW_ID
            ]
            
            logger.info(f"Filtered {len(flows)} flows down to {len(filtered_flows)} Better Board Game Search flows")
            return jsonify(filtered_flows)
        else:
            logger.error(f"Langflow API error: {response.status_code}")
            return jsonify({"error": "Failed to get flows from Langflow"}), response.status_code
            
    except Exception as e:
        logger.error(f"Error getting flows: {e}")
        return jsonify({"error": "Failed to get flows"}), 500

@app.route('/api/v1/run/<flow_id>', methods=['POST'])
def run_flow(flow_id):
    """Run a specific flow - only allow the target flow."""
    if flow_id != TARGET_FLOW_ID:
        return jsonify({
            "error": f"Flow {flow_id} not found. Only Better Board Game Search flow is available.",
            "available_flow_id": TARGET_FLOW_ID
        }), 404
    
    try:
        # Forward the request to Langflow
        response = requests.post(
            f"{LANGFLOW_BASE_URL}/api/v1/run/{flow_id}",
            headers=LANGFLOW_HEADERS,
            data=request.get_data(),
            timeout=60
        )
        
        # Return the response from Langflow
        return Response(
            response.content,
            status=response.status_code,
            headers=dict(response.headers)
        )
        
    except requests.exceptions.Timeout:
        logger.error("Request to Langflow timed out")
        return jsonify({"error": "Request timed out"}), 504
    except Exception as e:
        logger.error(f"Error running flow: {e}")
        return jsonify({"error": "Failed to run flow"}), 500

@app.route('/api/v1/flows/<flow_id>', methods=['GET'])
def get_flow(flow_id):
    """Get a specific flow - only allow the target flow."""
    if flow_id != TARGET_FLOW_ID:
        return jsonify({
            "error": f"Flow {flow_id} not found. Only Better Board Game Search flow is available.",
            "available_flow_id": TARGET_FLOW_ID
        }), 404
    
    try:
        response = requests.get(f"{LANGFLOW_BASE_URL}/api/v1/flows/{flow_id}", headers=LANGFLOW_HEADERS)
        return Response(
            response.content,
            status=response.status_code,
            headers=dict(response.headers)
        )
    except Exception as e:
        logger.error(f"Error getting flow: {e}")
        return jsonify({"error": "Failed to get flow"}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    try:
        # Check if Langflow is running
        response = requests.get(f"{LANGFLOW_BASE_URL}/api/v1/version", headers=LANGFLOW_HEADERS, timeout=5)
        if response.status_code == 200:
            return jsonify({
                "status": "healthy",
                "langflow_connected": True,
                "target_flow_id": TARGET_FLOW_ID,
                "target_flow_name": TARGET_FLOW_NAME
            })
        else:
            return jsonify({
                "status": "unhealthy",
                "langflow_connected": False,
                "error": "Langflow not responding"
            }), 503
    except Exception as e:
        return jsonify({
            "status": "unhealthy",
            "langflow_connected": False,
            "error": str(e)
        }), 503

@app.route('/', methods=['GET'])
def root():
    """Root endpoint with information about the API."""
    return jsonify({
        "name": "Better Board Game Search API",
        "description": "Custom API server exposing only the Better Board Game Search flow for personalized board game recommendations",
        "version": "1.0.0",
        "target_flow_id": TARGET_FLOW_ID,
        "target_flow_name": TARGET_FLOW_NAME,
        "endpoints": {
            "flows": "/api/v1/flows/",
            "run_flow": f"/api/v1/run/{TARGET_FLOW_ID}",
            "health": "/health",
            "version": "/api/v1/version"
        }
    })

if __name__ == '__main__':
    logger.info(f"Starting Better Board Game Search API Server")
    logger.info(f"Target Flow ID: {TARGET_FLOW_ID}")
    logger.info(f"Target Flow Name: {TARGET_FLOW_NAME}")
    logger.info(f"Langflow Base URL: {LANGFLOW_BASE_URL}")
    
    # Run the server
    app.run(host='0.0.0.0', port=5000, debug=True)
