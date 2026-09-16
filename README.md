# RAG Chatbot with OpenAI, Typesense, and PostgreSQL

A production-style Retrieval-Augmented Generation (RAG) chatbot built with Python, FastAPI, Dash, LlamaIndex, OpenAI, Typesense, and PostgreSQL. The app ingests documents, splits them into searchable chunks, retrieves relevant context, and answers user questions grounded in the source material.

## Features

- Document ingestion and chunking pipeline
- LlamaIndex-based RAG orchestration
- OpenAI chat and embedding models
- FastAPI backend for chat and document APIs
- Dash frontend for a web-based assistant UI
- PostgreSQL storage for relational metadata
- Typesense vector/search indexing for retrieval
- Environment-based configuration for secure deployment

## Architecture

- Frontend: Dash + Bootstrap
- Backend: FastAPI
- AI layer: LlamaIndex + OpenAI
- Vector search: Typesense
- Database: PostgreSQL
- Configuration: environment variables loaded from .env

## Project Structure

```bash
.
├── app/
│   ├── agents/
│   ├── api/
│   ├── dash_ui/
│   ├── db/
│   ├── ingestion/
│   ├── retrieval/
│   ├── config.py
│   └── logging_config.py
├── data/
├── scripts/
├── tests/
├── .env.example
├── .gitignore
├── docker-compose.yml
├── requirements.txt
├── conftest.py
├── README.md
└── pytest.ini
```

## Prerequisites

- Python 3.12+
- PostgreSQL running locally or in Docker
- Typesense running locally or via Docker
- OpenAI API key

## Quick start

1. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Configure environment variables

Copy the example environment file and update it with your real credentials:

```bash
cp .env.example .env
```

Then edit `.env` with your OpenAI key and database/search settings.

4. Start infrastructure services

```bash
docker compose up -d
```

This starts Typesense. PostgreSQL is expected to be available separately.

5. Initialize database tables

```bash
python scripts/create_db.py
```

6. Run the API backend

```bash
uvicorn app.api.main:app --reload --host 0.0.0.0 --port 8000
```

7. Run the Dash frontend

```bash
python app/dash_ui/app.py
```

The UI is typically available on:

- API: http://localhost:8000
- Dashboard: http://localhost:8050

## Environment configuration

The application reads configuration from a local `.env` file using `pydantic-settings`.

A sample file is provided at `.env.example`.

Important values include:

- `OPENAI_API_KEY`
- `OPENAI_CHAT_MODEL`
- `OPENAI_EMBEDDING_MODEL`
- `POSTGRES_*`
- `TYPESENSE_*`
- `APP_ENV`

Never commit your real `.env` file. It is already ignored in `.gitignore`.

## Common workflow

- Add documents to the ingestion layer or data folder
- Process and chunk documents
- Store metadata in PostgreSQL
- Index data in Typesense
- Query the assistant through the API or Dash UI

## Testing

```bash
pytest
```

## Notes

This project is structured for local development and experimentation with an RAG architecture using modern LLM tooling. It can be extended with additional ingestion formats, model providers, and deployment workflows.

## License

This project is provided for educational and development use. Add your preferred license if you plan to distribute or publish it publicly.
