-- migration_context_chunks.sql
-- pgvector chunking and embedding for context assets

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS context_chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    asset_id UUID NOT NULL REFERENCES context_assets(id) ON DELETE CASCADE,
    org_id UUID NOT NULL,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    embedding vector(1536),
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_chunks_asset ON context_chunks(asset_id);
CREATE INDEX IF NOT EXISTS idx_chunks_org ON context_chunks(org_id);
CREATE INDEX IF NOT EXISTS idx_chunks_embedding ON context_chunks
    USING hnsw (embedding vector_cosine_ops);

-- Semantic search function for agent RAG tools
CREATE OR REPLACE FUNCTION search_context_chunks(
    query_embedding vector(1536),
    match_org_id UUID,
    match_count INTEGER DEFAULT 5,
    filter_asset_ids UUID[] DEFAULT NULL
)
RETURNS TABLE (
    id UUID,
    asset_id UUID,
    asset_name TEXT,
    chunk_index INTEGER,
    content TEXT,
    similarity FLOAT
)
LANGUAGE sql STABLE
AS $$
    SELECT
        cc.id,
        cc.asset_id,
        ca.name AS asset_name,
        cc.chunk_index,
        cc.content,
        1 - (cc.embedding <=> query_embedding) AS similarity
    FROM context_chunks cc
    JOIN context_assets ca ON ca.id = cc.asset_id
    WHERE cc.org_id = match_org_id
      AND cc.embedding IS NOT NULL
      AND (filter_asset_ids IS NULL OR cc.asset_id = ANY(filter_asset_ids))
    ORDER BY cc.embedding <=> query_embedding
    LIMIT match_count;
$$;
