-- Cached content store — Supabase / Postgres backend (OPT-IN).
--
-- This is NOT applied automatically. The app is local-first; the cache lives in
-- data/cached/ unless you explicitly opt into the cloud backend. To enable it:
--
--   1. Install the Postgres driver (commented in requirements.txt):
--        pip install "psycopg[binary]>=3.2.0" "psycopg-pool>=3.2.0"
--   2. Apply this migration to your Supabase project (psql, the SQL editor, or
--        the Supabase MCP apply_migration tool). DDL is the only thing that goes
--        through the read-only MCP; runtime row writes use a direct DB URL.
--   3. Set in .env:
--        CACHED_STORE_BACKEND=supabase
--        SUPABASE_DB_URL=postgresql://...:6543/postgres   # transaction pooler
--      (CACHED_STORE_DB_URL overrides SUPABASE_DB_URL if you want a separate one.)
--
-- The app falls back to the local backend (with a stderr note) if the driver is
-- missing or the database is unreachable, so a misconfigured DB never blocks it.
-- Embeddings are NOT stored here — they stay local in data/vectors/library
-- (.npy). v1 keeps Q&A retrieval local even when content syncs to the cloud.

CREATE SCHEMA IF NOT EXISTS cached;

CREATE TABLE IF NOT EXISTS cached.cached_items (
    id            text PRIMARY KEY,        -- cache_<sha256(dedupe_key)[:16]>
    kind          text,                    -- 'url' | 'text' | 'file'
    source_url    text,                    -- '' for pasted/file items
    source_title  text,
    date          text,
    raw_text      text NOT NULL DEFAULT '',-- the canonical extracted content
    chars         int  DEFAULT 0,
    content_hash  text,                    -- sha256(raw_text); drift detection
    tags          jsonb DEFAULT '[]',
    note          text DEFAULT '',
    origin        text DEFAULT 'manual',   -- read_page | rag_ingest | kg_ingest | backfill | manual
    in_kg            boolean DEFAULT false,   -- projected into the knowledge graph?
    kg_chunk_ids     jsonb DEFAULT '[]',      -- chunk node ids created in overall.json
    kg_source_id     text DEFAULT '',
    kg_projected_hash text DEFAULT '',        -- content_hash at KG-projection time (drift flag)
    vectorized       boolean DEFAULT false,   -- projected into the vector library?
    vector_doc_id    text DEFAULT '',
    vector_projected_hash text DEFAULT '',    -- content_hash at vectorize time (drift flag)
    backend          text DEFAULT 'supabase',
    created_at    timestamptz DEFAULT now(),
    updated_at    timestamptz DEFAULT now()
);

CREATE INDEX IF NOT EXISTS cached_items_hash_idx ON cached.cached_items (content_hash);

-- The SupabaseCachedStore in cached_store.py runs the equivalent CREATE ... IF
-- NOT EXISTS on first connect, so applying this by hand is optional when the
-- direct DB URL has DDL privileges. It is kept here for review / read-only-MCP
-- setups where DDL is applied out of band.

-- ---------------------------------------------------------------------------
-- OPTIONAL cloud vector mirror (DEFERRED — only if you want cross-device Q&A).
-- Enabled by CACHE_VECTORS_BACKEND=supabase. Requires the pgvector extension.
-- Not used by v1; retrieval stays on the local HNSW index. Uncomment to use.
-- ---------------------------------------------------------------------------
-- CREATE EXTENSION IF NOT EXISTS vector;
--
-- CREATE TABLE IF NOT EXISTS cached.cached_chunk_vectors (
--     chunk_id            text PRIMARY KEY,
--     cache_id            text REFERENCES cached.cached_items(id) ON DELETE CASCADE,
--     doc_id              text, section_id text, title text, url text,
--     text                text, preview text, contextual_summary text,
--     position            int,
--     embedding           vector(1536)            -- text-embedding-3-small dim
-- );
-- CREATE TABLE IF NOT EXISTS cached.cached_section_vectors (
--     section_id          text PRIMARY KEY,
--     cache_id            text REFERENCES cached.cached_items(id) ON DELETE CASCADE,
--     doc_id              text, url text, title text, summary text, overview text,
--     section_index       int,
--     embedding           vector(1536)
-- );
-- CREATE INDEX IF NOT EXISTS cached_chunk_vec_idx
--     ON cached.cached_chunk_vectors USING hnsw (embedding vector_cosine_ops);
-- CREATE INDEX IF NOT EXISTS cached_section_vec_idx
--     ON cached.cached_section_vectors USING hnsw (embedding vector_cosine_ops);
