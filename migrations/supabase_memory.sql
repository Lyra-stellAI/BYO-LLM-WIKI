-- Memory layer — Supabase / Postgres backend (OPT-IN).
--
-- NOT applied automatically. The app is local-first; memory lives in
-- data/memory.json unless you opt into the cloud backend. To enable:
--
--   1. Install the Postgres driver (commented in requirements.txt):
--        pip install "psycopg[binary]>=3.2.0" "psycopg-pool>=3.2.0"
--   2. Apply this migration to your Supabase project (psql, SQL editor, or the
--        Supabase MCP apply_migration tool — DDL only; runtime row writes use a
--        direct DB URL, since the MCP server is read-only).
--   3. Set in .env:
--        MEMORY_BACKEND=supabase
--        SUPABASE_DB_URL=postgresql://...:6543/postgres   # transaction pooler
--      (MEMORY_DB_URL overrides SUPABASE_DB_URL for memory if set.)
--
-- Falls back to the local JSON store (with a stderr note) if the driver is
-- missing or the DB is unreachable, so a misconfigured DB never blocks recall.
--
-- One row per memory: the full record is stored as `doc` jsonb (the per-record
-- embedding is inline, as in the local store), so memory.py's Python recall /
-- dedup / supersede logic is unchanged and behavior is identical to local. The
-- SupabaseMemoryStore in memory.py runs the equivalent CREATE ... IF NOT EXISTS
-- on first connect, so applying this by hand is optional when the direct DB URL
-- has DDL privileges; it is kept here for read-only-MCP setups.

CREATE SCHEMA IF NOT EXISTS memory;

CREATE TABLE IF NOT EXISTS memory.memory_items (
    id          text PRIMARY KEY,          -- memory_<uuid>
    doc         jsonb NOT NULL,            -- the full memory record (embedding inline)
    kind        text,                      -- fact | answer | preference | gap | correction | observation
    superseded  boolean DEFAULT false,     -- true once a correction supersedes it
    updated_at  timestamptz DEFAULT now()
);

CREATE INDEX IF NOT EXISTS memory_items_kind_idx ON memory.memory_items (kind);

-- ---------------------------------------------------------------------------
-- OPTIONAL future optimization (DEFERRED): pgvector-accelerated recall.
-- v1 loads active records and scores them in Python (parity with local). To
-- push recall into SQL at scale, add a vector column + HNSW index and a
-- cosine-distance query path in memory.py. Not required for cloud durability.
-- ---------------------------------------------------------------------------
-- CREATE EXTENSION IF NOT EXISTS vector;
-- ALTER TABLE memory.memory_items ADD COLUMN IF NOT EXISTS embedding vector(1536);
-- CREATE INDEX IF NOT EXISTS memory_items_embedding_idx
--     ON memory.memory_items USING hnsw (embedding vector_cosine_ops);
