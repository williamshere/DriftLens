from fastapi.testclient import TestClient
from driftlens.api.app import create_app


def test_dashboard_and_health_endpoints():
    app = create_app()
    client = TestClient(app)

    # Health
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "version": "0.1.0"}

    # Dashboard root
    r_root = client.get("/")
    assert r_root.status_code == 200
    assert "DriftLens Dashboard" in r_root.text

    # Dashboard alias
    r_dash = client.get("/dashboard")
    assert r_dash.status_code == 200
    assert "DriftLens Dashboard" in r_dash.text


def test_api_network_and_drift_endpoints():
    import os
    import tempfile
    from pathlib import Path
    import numpy as np
    from driftlens.db.session import Database, TermCentroid

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        db_file = Path(td) / "test_api.db"
        db = Database(db_file)
        db.connect()
        db.init()

        # Insert test docs and centroids
        db.insert_document(doc_id="doc1", text="climate and policy resilience", slice_key="s1")
        db.insert_document(doc_id="doc2", text="climate and governance resilience", slice_key="s2")
        v1 = np.array([1.0, 0.0], dtype=np.float32)
        v2 = np.array([0.0, 1.0], dtype=np.float32)
        db.upsert_centroid(centroid=TermCentroid(term="climate", slice_key="s1", vector=v1, dim=2, doc_count=1, model="test"))
        db.upsert_centroid(centroid=TermCentroid(term="climate", slice_key="s2", vector=v2, dim=2, doc_count=1, model="test"))
        db.close()

        os.environ["DRIFTLENS_DB_PATH"] = str(db_file)
        app = create_app()
        client = TestClient(app)

        # Test network endpoint
        r_net = client.get("/api/network")
        assert r_net.status_code == 200
        net_data = r_net.json()
        assert "nodes" in net_data
        assert "edges" in net_data
        assert net_data["metrics"]["num_nodes"] > 0

        # Test drift endpoint
        r_drift = client.get("/api/drift?term=climate")
        assert r_drift.status_code == 200
        drift_data = r_drift.json()
        assert drift_data["term"] == "climate"
        assert len(drift_data["series"]) == 1
