"""Tests for graphify.storage — NeuG adapter layer (unified schema)."""
import json
import shutil
import tempfile
from pathlib import Path

import pytest

try:
    import neug
    _has_neug = True
except ImportError:
    _has_neug = False

pytestmark = pytest.mark.skipif(not _has_neug, reason="neug not installed")

FIXTURES = Path(__file__).parent / "fixtures"
EXTRACTION_JSON = FIXTURES / "extraction.json"


def _load_extraction() -> dict:
    return json.loads(EXTRACTION_JSON.read_text())


@pytest.fixture()
def tmp_db(tmp_path):
    db_path = str(tmp_path / "test.db")
    yield db_path


def _init(db_path):
    from graphify.storage import init_db, ensure_schema
    db, conn = init_db(db_path)
    ensure_schema(conn)
    return db, conn


def _close(db, conn):
    from graphify.storage import close_db
    close_db(db, conn)


def _query(conn, cypher):
    from graphify.storage import execute_cypher
    return execute_cypher(conn, cypher)


# --- init_db ---

def test_init_db_creates_tables(tmp_db):
    db, conn = _init(tmp_db)
    rows = _query(conn, "MATCH (n:node) RETURN count(n)")
    assert rows == [[0]]
    _close(db, conn)


# --- ingest_extraction: CREATE mode ---

def test_ingest_extraction_create_mode(tmp_db):
    from graphify.storage import ingest_extraction
    db, conn = _init(tmp_db)
    ext = _load_extraction()
    ingest_extraction(conn, ext, incremental=False)
    rows = _query(conn, "MATCH (n:node {type: 'code'}) RETURN n.id ORDER BY n.id")
    ids = sorted([r[0] for r in rows])
    assert "n_attention" in ids
    assert "n_transformer" in ids
    assert "n_layernorm" in ids
    edge_rows = _query(conn, "MATCH (a:node)-[e:edge {relation: 'contains'}]->(b:node) RETURN count(e)")
    assert edge_rows[0][0] == 2
    _close(db, conn)


# --- ingest_extraction: MERGE mode ---

def test_ingest_extraction_merge_mode(tmp_db):
    from graphify.storage import ingest_extraction
    db, conn = _init(tmp_db)
    ext = _load_extraction()
    ingest_extraction(conn, ext, incremental=False)
    ext["nodes"][0]["label"] = "TransformerV2"
    ingest_extraction(conn, ext, incremental=True)
    rows = _query(conn, "MATCH (n:node {type: 'code'}) WHERE n.id = 'n_transformer' RETURN n.label")
    assert rows[0][0] == "TransformerV2"
    count = _query(conn, "MATCH (n:node {type: 'code'}) RETURN count(n)")
    assert count[0][0] == 3
    _close(db, conn)


# --- file_type routing ---

def test_ingest_extraction_file_type_routing(tmp_db):
    from graphify.storage import ingest_extraction
    db, conn = _init(tmp_db)
    ext = _load_extraction()
    ingest_extraction(conn, ext, incremental=False)
    doc_rows = _query(conn, "MATCH (n:node {type: 'document'}) RETURN n.id")
    assert len(doc_rows) == 1
    assert doc_rows[0][0] == "n_concept_attn"
    _close(db, conn)


# --- prune_sources ---

def test_ingest_extraction_prune(tmp_db):
    from graphify.storage import ingest_extraction
    db, conn = _init(tmp_db)
    ext = _load_extraction()
    ingest_extraction(conn, ext, incremental=False)
    before = _query(conn, "MATCH (n:node {type: 'code'}) RETURN count(n)")[0][0]
    assert before == 3
    ingest_extraction(conn, ext, incremental=True, prune_sources=["model.py"])
    after_prune = _query(conn, "MATCH (n:node {type: 'code'}) RETURN count(n)")[0][0]
    assert after_prune == 3
    _close(db, conn)


# --- concepts ---

def test_ingest_concepts(tmp_db):
    from graphify.storage import ingest_extraction, ingest_concepts, get_concept_members
    db, conn = _init(tmp_db)
    ext = _load_extraction()
    ingest_extraction(conn, ext, incremental=False)
    concepts = [
        {"id": "concept_0", "name": "Transformer Module", "source": "leiden",
         "members": ["n_transformer", "n_attention"]},
        {"id": "concept_1", "name": "Normalization", "source": "leiden",
         "members": ["n_layernorm"]},
    ]
    ingest_concepts(conn, concepts)
    members = get_concept_members(conn)
    assert "concept_0" in members
    assert set(members["concept_0"]) == {"n_transformer", "n_attention"}
    assert "concept_1" in members
    assert set(members["concept_1"]) == {"n_layernorm"}
    _close(db, conn)


# --- execute_cypher ---

def test_execute_cypher(tmp_db):
    from graphify.storage import ingest_extraction
    db, conn = _init(tmp_db)
    ext = _load_extraction()
    ingest_extraction(conn, ext, incremental=False)
    rows = _query(conn, "MATCH (n:node {type: 'code'}) RETURN n.label ORDER BY n.id")
    labels = [r[0] for r in rows]
    assert "MultiHeadAttention" in labels
    assert "Transformer" in labels
    _close(db, conn)


def test_execute_cypher_bad_query(tmp_db):
    db, conn = _init(tmp_db)
    with pytest.raises(RuntimeError):
        _query(conn, "THIS IS NOT VALID CYPHER")
    _close(db, conn)


# --- end-to-end pipeline test ---

def test_full_pipeline_with_mock_leiden(tmp_db):
    """End-to-end test: extraction → NeuG → mock Leiden → concepts → delta detection."""
    import networkx as nx
    from graphify.storage import (
        ingest_extraction, ingest_concepts, get_concept_members,
        run_leiden_fallback, detect_concept_delta,
    )

    db, conn = _init(tmp_db)
    ext = _load_extraction()

    # Step 1: Ingest extraction into NeuG
    ingest_extraction(conn, ext, incremental=False)

    # Step 2: Build NetworkX graph from extraction (mock what extract does)
    G = nx.DiGraph()
    for node in ext["nodes"]:
        G.add_node(node["id"], label=node["label"], file_type=node["file_type"])
    for edge in ext["edges"]:
        src = edge.get("source") or edge.get("from")
        tgt = edge.get("target") or edge.get("to")
        G.add_edge(src, tgt, relation=edge["relation"])

    # Step 3: Run mock Leiden on NetworkX graph
    communities = run_leiden_fallback(G, resolution=1.0)
    assert len(communities) > 0
    assert all(isinstance(members, list) for members in communities.values())

    # Step 4: Write concepts to NeuG
    concepts = [
        {"id": f"concept_{cid}", "name": f"Community {cid}",
         "source": "leiden", "members": members}
        for cid, members in communities.items()
    ]
    ingest_concepts(conn, concepts)

    # Step 5: Verify concepts were written
    members_by_concept = get_concept_members(conn)
    assert len(members_by_concept) == len(communities)
    all_nodes = set()
    for members in members_by_concept.values():
        all_nodes.update(members)
    assert all_nodes == set(G.nodes())

    # Note: detect_concept_delta requires NeuG GDS Leiden (v0.1.3+)
    # which isn't available yet. The delta detection logic will be
    # tested once GDS is available.

    _close(db, conn)


# --- roundtrip consistency ---

def test_roundtrip_node_count(tmp_db):
    from graphify.storage import ingest_extraction
    db, conn = _init(tmp_db)
    ext = _load_extraction()
    ingest_extraction(conn, ext, incremental=False)
    rows = _query(conn, "MATCH (n:node) RETURN count(n)")
    assert rows[0][0] == len(ext["nodes"])
    _close(db, conn)
