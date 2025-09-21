# Vector Store RAG - Board Game Recommendations API

This custom API server exposes only your specific Vector Store RAG flow for board game recommendations, filtering out all other Langflow flows.

## 🎯 What This Does

- **Filters Langflow API**: Only exposes your specific flow (configured via `TARGET_FLOW_ID`)
- **Clean Interface**: When someone connects to `/api/v1/flows/`, they only see your board game recommendation flow
- **Secure Access**: Prevents access to other flows in your Langflow instance
- **Same Functionality**: All the same API endpoints work, just filtered

## 🚀 Quick Start

### Option 1: Local Only
```bash
./start_custom_api.sh
```

### Option 2: With ngrok (Public Access)
```bash
./start_custom_api_ngrok.sh
```

## 📋 API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /` | API information and available endpoints |
| `GET /health` | Health check and status |
| `GET /api/v1/version` | Langflow version info |
| `GET /api/v1/flows/` | **Returns only your Vector Store RAG flow** |
| `POST /api/v1/run/{flow_id}` | Run the flow (only accepts your flow ID) |

## 🧪 Testing

Test the API with:
```bash
python3 test_custom_api.py
```

Or test with a specific URL:
```bash
python3 test_custom_api.py http://your-ngrok-url.ngrok.io
```

## 🔧 Configuration

The custom API server is configured in `custom_api_server.py`:

- **Target Flow ID**: Configured via `TARGET_FLOW_ID` environment variable
- **Target Flow Name**: Configured via `TARGET_FLOW_NAME` environment variable
- **Langflow Backend**: `http://localhost:7861`
- **API Server Port**: `5000`

## 📁 Files Created

- `custom_api_server.py` - Main Flask API server
- `requirements_custom_api.txt` - Python dependencies
- `start_custom_api.sh` - Local startup script
- `start_custom_api_ngrok.sh` - Public access startup script
- `test_custom_api.py` - Test script
- `CUSTOM_API_README.md` - This documentation

## 🔄 How It Works

1. **Custom API Server** runs on port 5000
2. **Langflow** runs on port 7861 (backend)
3. **API Server** proxies requests to Langflow
4. **Filtering** happens at the API level:
   - `/api/v1/flows/` only returns Vector Store RAG flows
   - `/api/v1/run/{flow_id}` only accepts your specific flow ID
   - All other endpoints are blocked or filtered

## 🌍 Public Access

When using ngrok, your API will be available at:
```
https://your-ngrok-url.ngrok.io/api/v1/flows/
```

This URL will only show your Vector Store RAG flow, making it perfect for sharing with others who need board game recommendations.

## 🛡️ Security

- Only your specific flow is exposed
- Other flows in Langflow remain hidden
- Invalid flow IDs are rejected with 404 errors
- All requests are logged for monitoring

## 🔧 Troubleshooting

1. **API not starting**: Check if port 5000 is available
2. **Langflow connection failed**: Ensure Langflow is running on port 7861
3. **ngrok issues**: Check ngrok authentication and tunnel status
4. **Flow execution errors**: Verify your Vector Store RAG flow is working in Langflow UI

## 📊 Monitoring

- Check API health: `GET /health`
- View logs in terminal where you started the server
- Monitor ngrok dashboard: `http://localhost:4040`
