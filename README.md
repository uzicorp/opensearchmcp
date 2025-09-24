# OpenSeawrch MCP Demo

## Python Environment

This project uses uv and expects python v3.13

Add test dependencies:
```bash
uv add --dev pytest pyright ruff pytest-cov
```

Add runtime dependencies
```bash
uv add --dev pytest pyright ruff pytest-cov
```

Run Opensearch with docker compose:
```bash
docker-compose up --build
```

## Usage

**Creating a repo to do the following tasks:

1. Set up a Docker app with an OpenSearch instance
2. Download some PDFs from arXiv and index them in OpenSearch using HuggingFace BGE embeddings running locally - Done
3. Create a notebook with two or three different OpenSearch queries that will query the created index (by queries, I mean BM25 and dense vector search, with hybrid query as a nice to have!)
4. Additional points if embeddings and for any used LLMs are exposed through a LiteLLM proxy (**https://docs.litellm.ai/docs/simple_proxy**) running in the Docker app
5. Develop an MCP tool that accesses the OpenSearch index and performs searches based on the designed search techniques, then create a server using FastMCP (https://gofastmcp.com/getting-started/welcome)
6. Use the Agno framework (https://github.com/agno-agi/agno) to run this MCP server with stdio in an agent, then pass a simple user query and get a response from the LLM related to documents stored in the index. The same LiteLLM proxy can be used for the LLM with an open source model such as one from Hugging Face

If we could have this completed by next week Tuesday, that would be great 😊
Please let me know if there are any questions!**

## Notes

Remove docker containers to save space
```bash
docker system prune
```

When opensearch is running, you can access it at 172.18.0.2:9200