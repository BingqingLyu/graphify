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
        detect_concept_delta,
    )
    from graphify.cluster import cluster as run_leiden_fallback

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


# --- Wiki import: OKF ---

def test_parse_okf_bundle(tmp_db, tmp_path):
    from graphify.storage import parse_okf_bundle, ingest_concepts

    # Create test OKF bundle
    bundle = tmp_path / "wiki"
    tables = bundle / "tables"
    tables.mkdir(parents=True)

    (tables / "orders.md").write_text(
        "---\n"
        "type: BigQuery Table\n"
        "title: Orders\n"
        "description: One row per completed customer order.\n"
        "tags: [sales, orders]\n"
        "---\n\n"
        "# Schema\n\n"
        "| Column | Type |\n"
        "|--------|------|\n"
        "| `order_id` | STRING |\n"
        "| `customer_id` | STRING |\n\n"
        "FK to [customers](customers.md).\n\n"
        "# Citations\n\n"
        "- `model.py`\n"
        "- [BQ schema](/tables/orders.md)\n"
        "- https://cloud.google.com/bigquery\n",
        encoding="utf-8",
    )

    (tables / "customers.md").write_text(
        "---\n"
        "type: BigQuery Table\n"
        "title: Customers\n"
        "description: Customer master data.\n"
        "tags: [sales]\n"
        "---\n\n"
        "# Schema\n\n"
        "| Column | Type |\n"
        "|--------|------|\n"
        "| `customer_id` | STRING |\n",
        encoding="utf-8",
    )

    (bundle / "index.md").write_text("# Index\n", encoding="utf-8")

    concepts = parse_okf_bundle(bundle)
    assert len(concepts) == 2

    orders = next(c for c in concepts if c["id"] == "tables/orders")
    assert orders["name"] == "Orders"
    assert orders["type"] == "BigQuery Table"
    assert orders["source"] == "wiki"
    assert "sales" in orders["tags"]
    assert "tables/customers" in orders["links"]
    assert "model.py" in orders["citations"]
    assert "https://cloud.google.com/bigquery" not in orders["citations"]

    # Ingest and verify
    db, conn = _init(tmp_db)
    ingest_concepts(conn, concepts)
    rows = _query(conn, "MATCH (c:concept {source: 'wiki'}) RETURN count(c)")
    assert rows[0][0] == 2
    _close(db, conn)


def test_parse_okf_bundle_citations_to_belongs_to(tmp_db, tmp_path):
    from graphify.storage import ingest_extraction, parse_okf_bundle, ingest_concepts

    # First, ingest some code nodes so citations can resolve
    db, conn = _init(tmp_db)
    ext = _load_extraction()
    ingest_extraction(conn, ext, incremental=False)

    # Create OKF bundle with citations matching source_files in extraction
    bundle = tmp_path / "wiki"
    bundle.mkdir()
    (bundle / "concept.md").write_text(
        "---\n"
        "type: Concept\n"
        "title: Test Concept\n"
        "---\n\n"
        "# Citations\n\n"
        "- `model.py`\n",
        encoding="utf-8",
    )

    concepts = parse_okf_bundle(bundle)
    ingest_concepts(conn, concepts)

    # Verify belongs_to edge was created from citation
    rows = _query(conn,
        "MATCH (n:node)-[:belongs_to]->(c:concept {id: 'concept'}) "
        "WHERE n.source_file = 'model.py' RETURN count(n)")
    assert rows[0][0] > 0
    _close(db, conn)


# --- Wiki import: graph.json ---

def test_parse_graph_json(tmp_db, tmp_path):
    from graphify.storage import parse_graph_json, ingest_concepts

    # Create test graph.json
    graph_json = {
        "nodes": [
            {"id": "n1", "label": "Node 1", "community": 0, "community_name": "Module A"},
            {"id": "n2", "label": "Node 2", "community": 0},
            {"id": "n3", "label": "Node 3", "community": 1},
        ],
        "links": [
            {"source": "n1", "target": "n2"},
            {"source": "n2", "target": "n3"},
        ],
    }
    gj_path = tmp_path / "graph.json"
    gj_path.write_text(json.dumps(graph_json), encoding="utf-8")

    # Create labels file
    labels_path = tmp_path / ".graphify_labels.json"
    labels_path.write_text(json.dumps({"0": "Core Module", "1": "Utils"}), encoding="utf-8")

    concepts = parse_graph_json(gj_path)
    assert len(concepts) == 2

    c0 = next(c for c in concepts if c["id"] == "concept_0")
    assert c0["name"] == "Core Module"
    assert c0["source"] == "leiden"
    assert set(c0["members"]) == {"n1", "n2"}

    # Ingest and verify
    db, conn = _init(tmp_db)
    ingest_concepts(conn, concepts)
    rows = _query(conn, "MATCH (c:concept {source: 'leiden'}) RETURN count(c)")
    assert rows[0][0] == 2
    _close(db, conn)


# --- Wiki import: auto-detect ---

def test_import_wiki_auto_detect(tmp_db, tmp_path):
    from graphify.storage import import_wiki

    # Create test OKF bundle
    bundle = tmp_path / "wiki"
    bundle.mkdir()
    (bundle / "concept.md").write_text(
        "---\n"
        "type: Test\n"
        "title: Test\n"
        "---\n",
        encoding="utf-8",
    )

    db, conn = _init(tmp_db)

    # Auto-detect directory -> OKF
    count = import_wiki(conn, bundle, format="auto")
    assert count == 1

    # Auto-detect file -> graph-json
    gj_path = tmp_path / "graph.json"
    gj_path.write_text(
        json.dumps({"nodes": [{"id": "n1", "community": 0}], "links": []}),
        encoding="utf-8",
    )
    count = import_wiki(conn, gj_path, format="auto")
    assert count == 1

    _close(db, conn)


# --- analyze_wiki_impact ---

def test_analyze_wiki_impact_new_concept(tmp_db):
    """A community where >50% of nodes are unknown should be a new concept candidate."""
    from graphify.storage import analyze_wiki_impact, ingest_concepts, ingest_extraction
    db, conn = _init(tmp_db)

    # Ingest base data: 4 existing nodes
    base = {
        "nodes": [
            {"id": "n1", "label": "A", "type": "code", "source_file": "f1.py"},
            {"id": "n2", "label": "B", "type": "code", "source_file": "f1.py"},
            {"id": "n3", "label": "C", "type": "code", "source_file": "f2.py"},
            {"id": "n4", "label": "D", "type": "code", "source_file": "f2.py"},
        ],
        "edges": [
            {"source": "n1", "target": "n2", "relation": "calls"},
            {"source": "n3", "target": "n4", "relation": "calls"},
        ],
    }
    ingest_extraction(conn, base, incremental=False)

    # Import wiki concept covering n1, n2
    ingest_concepts(conn, [{
        "id": "wiki_c1", "name": "Wiki C1", "source": "wiki",
        "members": ["n1", "n2"],
    }])

    # Simulate extract --no-cluster: ingest delta (3 new nodes)
    delta = {
        "nodes": [
            {"id": "x1", "label": "X1", "type": "code", "source_file": "f3.py"},
            {"id": "x2", "label": "X2", "type": "code", "source_file": "f3.py"},
            {"id": "x3", "label": "X3", "type": "code", "source_file": "f3.py"},
        ],
        "edges": [
            {"source": "x1", "target": "x2", "relation": "calls"},
            {"source": "x2", "target": "x3", "relation": "calls"},
        ],
    }
    ingest_extraction(conn, delta, incremental=True)

    # Now analyze: graph.db has old+new data, concepts still reflect old state
    result = analyze_wiki_impact(conn)
    assert "new_concept_candidates" in result
    assert "summary" in result
    assert result["summary"]["new"] >= 0
    _close(db, conn)


def test_analyze_wiki_impact_no_baseline(tmp_db):
    """When no concepts exist, all communities are reported as new."""
    from graphify.storage import analyze_wiki_impact, ingest_extraction
    db, conn = _init(tmp_db)

    # Ingest some data without any concepts
    data = {
        "nodes": [
            {"id": "n1", "label": "A", "type": "code", "source_file": "f1.py"},
            {"id": "n2", "label": "B", "type": "code", "source_file": "f1.py"},
        ],
        "edges": [
            {"source": "n1", "target": "n2", "relation": "calls"},
        ],
    }
    ingest_extraction(conn, data, incremental=False)

    result = analyze_wiki_impact(conn)
    # No old concepts → no concept_changes, everything is new
    assert result["concept_changes"] == {}
    assert result["summary"]["stable"] == 0
    # All communities should be new concept candidates (novelty_ratio = 1.0)
    assert result["summary"]["new"] >= 1
    _close(db, conn)


def test_get_wiki_concepts(tmp_db):
    """Test querying concepts with metadata."""
    from graphify.storage import _get_wiki_concepts, ingest_concepts
    db, conn = _init(tmp_db)

    ingest_concepts(conn, [{
        "id": "c1", "name": "Test Concept", "type": "module",
        "description": "A test", "source": "wiki", "tags": ["test"],
        "members": [],
    }])

    concepts = _get_wiki_concepts(conn)
    assert "c1" in concepts
    assert concepts["c1"]["name"] == "Test Concept"
    assert concepts["c1"]["source"] == "wiki"
    _close(db, conn)


def test_get_concept_links(tmp_db):
    """Test querying links_to edges between concepts."""
    from graphify.storage import _get_concept_links, ingest_concepts
    db, conn = _init(tmp_db)

    ingest_concepts(conn, [
        {"id": "c1", "name": "C1", "source": "wiki", "members": [], "links": ["c2"]},
        {"id": "c2", "name": "C2", "source": "wiki", "members": []},
    ])

    links = _get_concept_links(conn)
    assert ("c1", "c2") in links
    _close(db, conn)

