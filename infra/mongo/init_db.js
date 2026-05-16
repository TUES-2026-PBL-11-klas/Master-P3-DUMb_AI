const dbName = "dumb_ai";
const database = db.getSiblingDB(dbName);

const collections = [
  "users",
  "documents",
  "document_chunks",
  "queries",
  "schema_versions",
];

for (const collectionName of collections) {
  if (!database.getCollectionNames().includes(collectionName)) {
    database.createCollection(collectionName);
  }
}

database.users.createIndex({ username: 1 }, { unique: true });

database.documents.createIndex({ user_id: 1 });
database.documents.createIndex({ uploaded_at: -1 });

database.document_chunks.createIndex({ doc_id: 1 });

database.queries.createIndex({ user_id: 1 });
database.queries.createIndex({ created_at: -1 });

database.schema_versions.updateOne(
  { _id: "schema" },
  {
    $set: {
      version: 1,
      updated_at: new Date(),
      description: "Initial MongoDB document model for DUMb_AI",
    },
  },
  { upsert: true },
);

const searchIndexName = "chunk_embedding_vector_index";
const existingSearchIndexes = database.document_chunks
  .getSearchIndexes()
  .map((index) => index.name);

if (!existingSearchIndexes.includes(searchIndexName)) {
  database.document_chunks.createSearchIndex(
    searchIndexName,
    "vectorSearch",
    {
      fields: [
        {
          type: "vector",
          path: "embedding",
          numDimensions: 1024,
          similarity: "cosine",
        },
        {
          type: "filter",
          path: "doc_id",
        },
      ],
    },
  );
}

print(`Initialized MongoDB database '${dbName}'.`);
print(`Collections: ${database.getCollectionNames().sort().join(", ")}`);
print(`Search indexes: ${database.document_chunks
  .getSearchIndexes()
  .map((index) => `${index.name}:${index.status}`)
  .join(", ")}`);
