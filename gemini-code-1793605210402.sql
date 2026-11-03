PRAGMA journal_mode = WAL;

CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    author TEXT,
    timestamp DATETIME NOT NULL,
    time_slice TEXT NOT NULL,       -- e.g., '2024-Q1', '2024-03'
    community TEXT NOT NULL,        -- e.g., subreddit or forum tag
    raw_text TEXT NOT NULL,
    cleaned_text TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS embeddings (
    doc_id TEXT PRIMARY KEY,
    vector BLOB NOT NULL,           -- Compact float32 numpy array bytes
    FOREIGN KEY(doc_id) REFERENCES documents(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS term_centroids (
    term TEXT NOT NULL,
    time_slice TEXT NOT NULL,
    community TEXT NOT NULL,
    centroid BLOB NOT NULL,         -- Mean vector of matching contexts
    sample_count INTEGER NOT NULL,
    PRIMARY KEY (term, time_slice, community)
);

CREATE INDEX IF NOT EXISTS idx_docs_slice ON documents(time_slice);
CREATE INDEX IF NOT EXISTS idx_docs_community ON documents(community);