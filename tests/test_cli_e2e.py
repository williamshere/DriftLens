"""Focused regression for CLI ingest -> embeddings -> drift wiring.

Covers the bug where cli.py passed positional conn as doc argument.
"""
import tempfile
from pathlib import Path

from click.testing import CliRunner

from driftlens.cli import app
from driftlens.db.session import Database


def test_cli_ingest_process_drift_roundtrip():
    runner = CliRunner()
    with runner.isolated_filesystem():
        # create corpus
        Path("corpus.csv").write_text(
            "text,timestamp,author,community\n"
            "climate policy economy,2023-01-10,alice,sci\n"
            "identity labor fairness,2023-01-15,bob,sci\n"
            "climate identity labor,2023-02-01,carol,sci\n",
            encoding="utf-8",
        )
        db_path = "e2e.db"
        r1 = runner.invoke(app, ["--db-path", db_path, "ingest", "--file", "corpus.csv", "--slice-by", "timestamp"])
        assert r1.exit_code == 0, r1.output
        assert Database(db_path).count_documents() == 3

        r2 = runner.invoke(app, ["--db-path", db_path, "process-embeddings"])
        assert r2.exit_code == 0, r2.output
        assert Database(db_path).count_embeddings() == 3

        r3 = runner.invoke(app, ["--db-path", db_path, "compute-drift", "--metric", "cosine"])
        assert r3.exit_code == 0, r3.output
        assert "drift_magnitude" in r3.output

        r4 = runner.invoke(app, ["--db-path", db_path, "compute-drift", "--metric", "euclidean", "--top-k", "1"])
        assert r4.exit_code == 0, r4.output
