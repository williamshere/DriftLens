import sys
import subprocess
from pathlib import Path

# Auto-relaunch inside project virtual environment if invoked with system Python
_venv_python = Path(__file__).resolve().parent / ".venv" / "Scripts" / "python.exe"
if _venv_python.exists() and Path(sys.executable).resolve() != _venv_python.resolve():
    sys.exit(subprocess.call([str(_venv_python), __file__] + sys.argv[1:]))

import csv
from click.testing import CliRunner
from driftlens.cli import app
from driftlens.db.session import Database
from driftlens.analytics.network import build_cooccurrence_graph, export_for_d3

corpus = [
    {'text':'climate policy and economic labor markets shape community trust','timestamp':'2023-01-05','author':'alice','community':'sci'},
    {'text':'identity fairness and community in labor negotiations','timestamp':'2023-01-18','author':'bob','community':'sci'},
    {'text':'economic policy governance and climate action require fairness','timestamp':'2023-02-08','author':'alice','community':'sci'},
    {'text':'labor identity and community building for climate resilience','timestamp':'2023-02-20','author':'carol','community':'sci'},
    {'text':'fairness in economic policy shapes community outcomes','timestamp':'2023-03-02','author':'bob','community':'sci'},
    {'text':'policy labor and identity intersect in climate discourse','timestamp':'2023-03-15','author':'dave','community':'sci'},
]
Path('example_corpus.csv').write_text('',encoding='utf-8')
with open('example_corpus.csv','w',newline='',encoding='utf-8') as f:
    w=csv.DictWriter(f, fieldnames=['text','timestamp','author','community'])
    w.writeheader()
    for r in corpus: w.writerow(r)

import os
for p in ['example.db']:
    try: os.remove(p)
    except: pass

runner = CliRunner()
print('=== 1. INGEST (CSV, slice-by timestamp) ===')
r = runner.invoke(app, ['--db-path','example.db','ingest','--file','example_corpus.csv','--slice-by','timestamp'])
print(r.output)
print('DB count', Database('example.db').count_documents())
for d in Database('example.db').list_documents():
    print(f"  {d.id[:8]} | {d.slice_key} | {d.text[:45]}")

print('\n=== 2. PROCESS-EMBEDDINGS (mock deterministic) ===')
r = runner.invoke(app, ['--db-path','example.db','process-embeddings'])
print(r.output)
print('embeddings', Database('example.db').count_embeddings())

print('\n=== 3. COMPUTE-DRIFT (cosine, all terms) ===')
r = runner.invoke(app, ['--db-path','example.db','compute-drift','--metric','cosine'])
print(r.output[:2500])

print('\n=== 4. COMPUTE-DRIFT (euclidean top-3) ===')
r = runner.invoke(app, ['--db-path','example.db','compute-drift','--metric','euclidean','--top-k','3'])
print(r.output[:1500])

print('\n=== 5. COMPUTE-DRIFT (filter to climate,policy) ===')
r = runner.invoke(app, ['--db-path','example.db','compute-drift','--terms','climate,policy','--metric','cosine'])
print(r.output)

print('\n=== 6. CO-OCCURRENCE NETWORK (all docs) ===')
db = Database('example.db')
G = build_cooccurrence_graph(db.list_documents())
payload = export_for_d3(G)
print(f"nodes={len(payload['nodes'])} edges={len(payload['edges'])}")
for n in sorted(payload['nodes'], key=lambda x: -x['degree_centrality'])[:5]:
    print(f"  {n['id']:12s} freq={n['frequency']} deg={n['degree_centrality']:.2f} bet={n['betweenness_centrality']:.2f} comm={n['community']}")
print('edges sample', payload['edges'][:3])

print('\n=== 7. DRIFT via Python API ===')
from driftlens.analytics.drift import compute_drift_series
centroids = {c.slice_key: c.vector for c in db.list_centroids(term='climate')}
print('climate slices', sorted(centroids.keys()))
if centroids:
    series = compute_drift_series(centroids, metric='cosine')
    for s in series: print(f"  {s['from_slice']} -> {s['to_slice']}: drift={s['drift']:.4f} sim={s['similarity']:.4f}")

print('\nExample complete. DB=example.db, corpus=example_corpus.csv')
print('Try: driftlens --db-path example.db serve --port 8000  (then curl http://127.0.0.1:8000/api/drift?term=climate)')
