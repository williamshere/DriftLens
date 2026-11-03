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

