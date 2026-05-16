# Master-P3-DUMb_AI

## Local MongoDB

The project uses MongoDB Atlas Local for local development because the RAG
pipeline needs MongoDB Vector Search.

Start the database:

```bash
docker compose up -d mongodb-atlas-local
```

Initialize collections, normal indexes, schema versioning, and the vector
search index:

```bash
mongosh "mongodb://localhost:27018" infra/mongo/init_db.js
```

Connection string:

```txt
mongodb://localhost:27018
```

More details are in `infra/mongo/README.md`.
