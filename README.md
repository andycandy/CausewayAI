# CausewayAI Causal Engine
## Overview
CausewayAI is a backend system designed for causal analysis and retrieval-augmented generation (RAG) using FastAPI, Qdrant, and NetworkX. It leverages Google's Gemini models and `embeddinggemma` for advanced query understanding and response generation.

## Features
- **Graph Strategy**: Uses a knowledge graph to traverse causal logic and retrieve relevant calls.
- **Filter Strategy**: Uses metadata filtering to find relevant conversations based on domain, topic, and outcome.
- **Streaming Responses**: Returns Server-Sent Events (SSE) for real-time interaction.

## Prerequisites
- Python 3.10+
- Docker & Docker Compose (optional but recommended)
- Google API Key (for Gemini)
- Hugging Face Token (for `embeddinggemma`)

## Setup

### Environment Variables
Create a `.env` file in the root directory:
```env
GOOGLE_API_KEY=your_google_api_key
HF_TOKEN=your_hugging_face_token
```

### Running Locally
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the server:
   ```bash
   uvicorn main:app --reload
   ```
   The API will be available at `http://localhost:8000`.

### Running with Docker
1. Build and run the container:
   ```bash
   docker-compose up --build
   ```
   The API will be available at `http://localhost:8000`.

## API Endpoints
- **GET /health**: Health check.
- **POST /chat**: Main chat endpoint.
  - **Payload**:
    ```json
    {
      "query": "Why are customers churning?",
      "history": [],
      "model_type": "graph"  // or "filter"
    }
    ```
  - **Response**: Streaming SSE events (`status`, `concepts`, `sources`, `token`, `error`).

## Testing
Use the provided client script to test the engine:
```bash
python client_test.py "Why are customers churning?" --model graph
```
