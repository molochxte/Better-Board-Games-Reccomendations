# Better Board Game Recommendations

A Langflow-based AI system for personalized board game recommendations using vector search and RAG (Retrieval Augmented Generation).

## 🎯 Features

- **AI-Powered Recommendations**: Uses Langflow with custom components for intelligent board game suggestions
- **Vector Search**: Leverages Neon Database with pgvector for semantic search
- **Custom API**: Filtered API that exposes only your specific recommendation flow
- **Public Access**: Easy deployment with ngrok for external access
- **Modern Frontend**: Clean web interface for testing recommendations

## 🏗️ Architecture

```
Frontend (HTML/JS) → Custom API Server (Flask) → Langflow → Neon Database
```

## 📋 Prerequisites

- Python 3.8+
- Langflow
- Neon Database account
- OpenAI API key
- ngrok (for public access)

## 🚀 Quick Start

### 1. Clone and Setup

```bash
git clone <your-repo-url>
cd Better-Board-Games-Reccomendations
pip install -r requirements.txt
pip install -r requirements_custom_api.txt
```

### 2. Configure Environment

Copy the example environment file and fill in your values:

```bash
cp env.example .env
```

Edit `.env` with your configuration:

```env
# Langflow Configuration
LANGFLOW_BASE_URL=http://localhost:7861
TARGET_FLOW_ID=your-flow-id-here
TARGET_FLOW_NAME=Better Board Game Search

# Neon Database Configuration
NEON_CONNECTION_STRING=postgresql://username:password@ep-xxxxx.us-east-1.aws.neon.tech/dbname?sslmode=require
NEON_COLLECTION_NAME=board_games

# OpenAI Configuration
OPENAI_API_KEY=sk-your-openai-api-key-here

# ngrok Configuration (optional)
NGROK_AUTHTOKEN=your-ngrok-authtoken-here
```

### 3. Setup Neon Database

1. Create a Neon database at [console.neon.tech](https://console.neon.tech)
2. Enable the pgvector extension
3. Get your connection string from the Neon console
4. Update `NEON_CONNECTION_STRING` in your `.env` file

### 4. Setup Data

1. **Add Board Game Data**: Place your CSV datasets in the `data/` directory
2. **Data Format**: See `data/README.md` for required column format
3. **Sample Data**: You can use BoardGameGeek data or create your own

### 5. Setup Langflow

1. Install Langflow: `pip install langflow`
2. Create your board game recommendation flow in Langflow
3. Configure your data source to point to `data/your_dataset.csv`
4. Note your flow ID and update `TARGET_FLOW_ID` in `.env`

### 6. Run the System

#### Local Development
```bash
./start_custom_api.sh
```

#### Public Access (with ngrok)
```bash
./start_custom_api_ngrok.sh
```

### 7. Test the System

1. Open `frontend_custom_api.html` in your browser
2. Enter your API URL (local: `http://localhost:5000` or ngrok URL)
3. Test the connection and get recommendations!

## 📁 Project Structure

```
Better-Board-Games-Reccomendations/
├── CustomComponents/
│   └── neon_vector.py          # Custom Langflow component for Neon Database
├── data/                       # Board game datasets (excluded from Git)
│   └── README.md              # Data directory documentation
├── custom_api_server.py        # Flask API server (filters Langflow)
├── frontend_custom_api.html    # Web interface
├── start_custom_api.sh         # Local startup script
├── start_custom_api_ngrok.sh   # Public startup script
├── requirements.txt            # Main dependencies
├── requirements_custom_api.txt # API server dependencies
├── env.example                 # Environment variables template
├── .gitignore                  # Git ignore file
└── README.md                   # This file
```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `LANGFLOW_BASE_URL` | Langflow server URL | `http://localhost:7861` |
| `TARGET_FLOW_ID` | Your Langflow flow ID | `your-flow-id-here` |
| `TARGET_FLOW_NAME` | Your flow name | `Better Board Game Search` |
| `NEON_CONNECTION_STRING` | Neon database connection | Required |
| `OPENAI_API_KEY` | OpenAI API key | Required |
| `NGROK_AUTHTOKEN` | ngrok authentication token | Optional |

### Custom Langflow Component

The `CustomComponents/neon_vector.py` file contains a custom Langflow component that:

- Connects to Neon Database with pgvector
- Handles document ingestion and vector search
- Provides board game recommendation functionality
- Supports batch processing for large datasets

## 🌐 API Endpoints

### Custom API Server

- `GET /health` - Health check
- `GET /api/v1/flows/` - List available flows (filtered)
- `POST /api/v1/run/{flow_id}` - Execute flow
- `GET /` - API information

### Example Usage

```bash
# Health check
curl http://localhost:5000/health

# Get flows
curl http://localhost:5000/api/v1/flows/

# Run recommendation
curl -X POST http://localhost:5000/api/v1/run/your-flow-id \
  -H "Content-Type: application/json" \
  -d '{"input_value": "I want a cooperative game for 4 players"}'
```

## 🛠️ Development

### Adding New Features

1. Modify the custom Langflow component in `CustomComponents/neon_vector.py`
2. Update the API server in `custom_api_server.py` if needed
3. Test with the frontend interface

### Database Schema

The system expects a board games dataset with columns:
- Game name, year, players, duration, rating
- Categories, mechanics, descriptions
- BGG (BoardGameGeek) links and IDs

## 📊 Data Sources

- BoardGameGeek (BGG) dataset
- Game descriptions and metadata
- User ratings and categories

## 🔒 Security

- API keys are stored in environment variables
- Custom API filters access to only your specific flow
- ngrok provides secure tunneling for public access
- All sensitive data is excluded from version control

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [Langflow](https://github.com/langflow-ai/langflow) for the AI workflow platform
- [Neon](https://neon.tech) for the PostgreSQL database service
- [OpenAI](https://openai.com) for the language models
- [BoardGameGeek](https://boardgamegeek.com) for the game data

## 📞 Support

For issues and questions:
1. Check the [Issues](https://github.com/your-username/Better-Board-Games-Reccomendations/issues) page
2. Review the documentation
3. Create a new issue with detailed information

---

**Happy Gaming! 🎲**