-- DriftLens SQLite schema
-- Enable WAL for concurrent read/write throughput.
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;
PRAGMA synchronous=NORMAL;

-- ------------------------------------------------------------------
-- Documents
-- ------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS documents (
    id              TEXT PRIMARY KEY,
    text            TEXT NOT NULL,
    title           TEXT,
    slice_key       TEXT,
    meta_json       TEXT,               -- arbitrary JSON metadata
    created_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    updated_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE INDEX IF NOT EXISTS idx_documents_slice_key ON documents(slice_key);
CREATE INDEX IF NOT EXISTS idx_documents_created_at ON documents(created_at);

-- Keep updated_at fresh on row updates.
CREATE TRIGGER IF NOT EXISTS trg_documents_updated_at
AFTER UPDATE ON documents FOR EACH ROW
BEGIN
    UPDATE documents SET updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now') WHERE id = NEW.id;
END;

-- ------------------------------------------------------------------
-- Embeddings  (one row per document; vector stored as BLOB float32)
-- ------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS embeddings (
    doc_id          TEXT PRIMARY KEY,
    vector          BLOB NOT NULL,      -- float32 bytes via numpy.tobytes()
    dim             INTEGER NOT NULL,
    model           TEXT NOT NULL,
    created_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    FOREIGN KEY (doc_id) REFERENCES documents(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_embeddings_model ON embeddings(model);
CREATE INDEX IF NOT EXISTS idx_embeddings_dim ON embeddings(dim);

-- ------------------------------------------------------------------
-- Term centroids  (aggregated vectors per term + slice)
-- ------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS term_centroids (
    term            TEXT NOT NULL,
    slice_key       TEXT NOT NULL,
    vector          BLOB NOT NULL,      -- float32 centroid bytes
    dim             INTEGER NOT NULL,
    doc_count       INTEGER NOT NULL DEFAULT 0,
    model           TEXT NOT NULL,
    created_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    updated_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    PRIMARY KEY (term, slice_key)
);

CREATE INDEX IF NOT EXISTS idx_term_centroids_term ON term_centroids(term);
CREATE INDEX IF NOT EXISTS idx_term_centroids_slice_key ON term_centroids(slice_key);
CREATE INDEX IF NOT EXISTS idx_term_centroids_model ON term_centroids(model);

CREATE TRIGGER IF NOT EXISTS trg_term_centroids_updated_at
AFTER UPDATE ON term_centroids FOR EACH ROW
BEGIN
    UPDATE term_centroids SET updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
    WHERE term = NEW.term AND slice_key = NEW.slice_key;
END;
