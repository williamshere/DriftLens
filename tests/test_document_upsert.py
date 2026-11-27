import numpy as np

from driftlens.db.session import Database, TermCentroid


def test_reingesting_unchanged_document_preserves_derived_data(tmp_path):
    db = Database(tmp_path / "driftlens.db")
    db.init_db()
    db.insert_document(doc_id="doc-1", text="climate policy", slice_key="2024")
    db.insert_embedding(doc_id="doc-1", vector=np.array([1.0, 0.0]))
    db.upsert_centroid(
        centroid=TermCentroid(
            term="climate",
            slice_key="2024",
            vector=np.array([1.0, 0.0]),
            dim=2,
            doc_count=1,
            model="test",
        )
    )

    db.insert_documents(
        [{"id": "doc-1", "text": "climate policy", "slice_key": "2024"}]
    )

    assert db.count_embeddings() == 1
    assert db.count_centroids() == 1


def test_changing_document_invalidates_embedding_and_centroids(tmp_path):
    db = Database(tmp_path / "driftlens.db")
    db.init_db()
    db.insert_document(doc_id="doc-1", text="climate policy", slice_key="2024")
    db.insert_embedding(doc_id="doc-1", vector=np.array([1.0, 0.0]))
    db.upsert_centroid(
        centroid=TermCentroid(
            term="climate",
            slice_key="2024",
            vector=np.array([1.0, 0.0]),
            dim=2,
            doc_count=1,
            model="test",
        )
    )

    db.insert_document(doc_id="doc-1", text="new policy", slice_key="2025")

    assert db.count_embeddings() == 0
    assert db.count_centroids() == 0
