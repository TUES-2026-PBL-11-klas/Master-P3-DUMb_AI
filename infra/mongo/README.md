# MongoDB Atlas Local

This directory contains the local MongoDB setup for DUMb_AI.


## Start MongoDB

From the repository root:

```bash
docker compose up -d mongodb-atlas-local
```

The container exposes MongoDB on:

```txt
mongodb://localhost:27018
```

## Initialize The Database

Run:

```bash
mongosh "mongodb://localhost:27018" infra/mongo/init_db.js
```

The script creates:

- `users`
- `documents`
- `document_chunks`
- `queries`

It also creates the MongoDB Vector Search index:

```txt
chunk_embedding_vector_index
```

on:

```txt
document_chunks.embedding
```

## Verify

```bash
mongosh "mongodb://localhost:27018"
```

Then:

```javascript
use dumb_ai
show collections
db.document_chunks.getSearchIndexes()
```

The vector index should eventually show:

```txt
status: READY
queryable: true
```
