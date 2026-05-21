FROM pgvector/pgvector:pg16

COPY vectorstore/schema.sql /docker-entrypoint-initdb.d/schema.sql
