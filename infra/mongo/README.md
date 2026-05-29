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
- `schema_versions`

## Collections

The schema follows the names used in the Python domain code. If a field already
exists in code, the database documentation keeps that exact name.

### `users`

Stores local user accounts.

Fields:

- `_id`
- `username`
- `password_hash`
- `created_at`

Indexes:

- unique index on `username`

### `documents`

Stores one uploaded/parsed document per record. The raw parsed text is stored in
`content`, matching `Document.content` in `services/shared/domain.py`.

Fields:

- `_id`
- `user_id`
- `content`
- `filename`
- `uploaded_at`
- `status`
- `error_message`
- `schema_version`

Indexes:

- `user_id`
- `uploaded_at`

### `document_chunks`

Stores the chunks produced from a document and their embeddings. A chunk is
identified logically by the composite key `(doc_id, position)`: `doc_id` points
to the document, and `position` is the ordered chunk number inside that document.

Fields:

- `_id`
- `doc_id`
- `user_id`
- `position`
- `text`
- `embedding`
- `token_count`
- `metadata`
- `created_at`
- `schema_version`

Indexes:

- `doc_id`
- compound index on `user_id` and `doc_id`
- unique compound index on `doc_id` and `position`
- vector search index on `embedding`

### `queries`

Stores user questions and generated answers.

Fields:

- `_id`
- `user_id`
- `question`
- `answer`
- `source_chunk_ids`
- `created_at`
- `model`
- `embedding_model`

Indexes:

- `user_id`
- `created_at`

### `schema_versions`

Stores the current database schema version used by the local initialization
script.

Fields:

- `_id`
- `version`
- `updated_at`
- `description`

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
db.schema_versions.find()
```

The vector index should eventually show:

```txt
status: READY
queryable: true
```
