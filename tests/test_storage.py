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


def test_ingest_extraction_incremental_after_leiden_column(tmp_db):
    from graphify.storage import ingest_extraction
    db, conn = _init(tmp_db)
    ext = _load_extraction()
    ingest_extraction(conn, ext, incremental=False)
    conn.execute("MATCH (n:node) SET n.leiden_comm = 10")
    ext["nodes"][0]["label"] = "TransformerV2"
    ingest_extraction(conn, ext, incremental=True)
    rows = _query(
        conn,
        "MATCH (n:node) WHERE n.id = 'n_transformer' RETURN n.label, n.leiden_comm",
    )
    assert rows[0][0] == "TransformerV2"
    assert rows[0][1] >= 0
    _close(db, conn)


def test_ingest_extraction_incremental_prepares_nonnegative_leiden_comm(tmp_db):
    from graphify.storage import ingest_extraction
    db, conn = _init(tmp_db)
    ext = _load_extraction()
    ingest_extraction(conn, ext, incremental=False)
    conn.execute("MATCH (n:node) SET n.leiden_comm = 10")
    ext["nodes"][0]["label"] = "TransformerV2"
    ingest_extraction(conn, ext, incremental=True)
    rows = _query(
        conn,
        "MATCH (n:node) RETURN min(n.leiden_comm), count(n)",
    )
    assert rows[0][0] >= 0
    assert rows[0][1] >= 3
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

    # Base: two disconnected pairs → Leiden produces 2 communities
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

    # Set leiden_comm: n1/n2 → community 0, n3/n4 → community 1
    conn.execute("MATCH (n:node) WHERE n.id IN ['n1','n2'] SET n.leiden_comm = 0")
    conn.execute("MATCH (n:node) WHERE n.id IN ['n3','n4'] SET n.leiden_comm = 1")

    # Wiki concept covering n1, n2
    ingest_concepts(conn, [{
        "id": "wiki_c1", "name": "Wiki C1", "source": "wiki",
        "members": ["n1", "n2"],
    }])

    # Delta: 3 new nodes forming a disconnected component → Leiden creates a 3rd community
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

    # Analyze with real Leiden — no mock
    result = analyze_wiki_impact(conn)
    assert "new_concept_candidates" in result
    assert "summary" in result
    # The x1/x2/x3 community should be detected as a new concept candidate
    assert result["summary"]["new"] >= 1
    new_cands = result["new_concept_candidates"]
    new_nodes = set()
    for cand in new_cands:
        new_nodes.update(cand["members"])
    assert {"x1", "x2", "x3"}.issubset(new_nodes)
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


def test_analyze_wiki_impact_tracks_delta_and_weak_links(tmp_db):
    """Real DB + real Leiden: merge, growth, delta, and weak links."""
    from graphify.storage import analyze_wiki_impact, ingest_concepts, ingest_extraction
    db, conn = _init(tmp_db)

    # Base graph: 5 old nodes.
    #   old_a <-> old_b <-> drift  (connected triplet)
    #   old_c                      (isolated)
    #   old_d                      (isolated)
    base = {
        "nodes": [
            {"id": "old_a", "label": "A", "type": "code", "source_file": "f1.py"},
            {"id": "old_b", "label": "B", "type": "code", "source_file": "f1.py"},
            {"id": "drift", "label": "D", "type": "code", "source_file": "f2.py"},
            {"id": "old_c", "label": "C", "type": "code", "source_file": "f2.py"},
            {"id": "old_d", "label": "E", "type": "code", "source_file": "f3.py"},
        ],
        "edges": [
            {"source": "old_a", "target": "old_b", "relation": "calls"},
            {"source": "old_b", "target": "drift", "relation": "calls"},
        ],
    }
    ingest_extraction(conn, base, incremental=False)

    # Set leiden_comm: old_a/old_b=0, drift=1, old_c=2, old_d=3
    conn.execute("MATCH (n:node) WHERE n.id IN ['old_a','old_b'] SET n.leiden_comm = 0")
    conn.execute("MATCH (n:node) WHERE n.id = 'drift' SET n.leiden_comm = 1")
    conn.execute("MATCH (n:node) WHERE n.id = 'old_c' SET n.leiden_comm = 2")
    conn.execute("MATCH (n:node) WHERE n.id = 'old_d' SET n.leiden_comm = 3")

    # Concepts: c1={old_a,old_b}, c2={drift}, c3={old_c}, c4={old_d}
    ingest_concepts(conn, [
        {"id": "c1", "name": "Concept One", "source": "wiki", "members": ["old_a", "old_b"]},
        {"id": "c2", "name": "Concept Two", "source": "wiki", "members": ["drift"]},
        {"id": "c3", "name": "Concept Three", "source": "wiki", "members": ["old_c"]},
        {"id": "c4", "name": "Concept Four", "source": "wiki", "members": ["old_d"]},
    ])

    # Delta: delta1->old_a, delta2->old_b (join the triplet community)
    #        delta3->old_d (joins old_d)
    delta = {
        "nodes": [
            {"id": "delta1", "label": "D1", "type": "code", "source_file": "f4.py"},
            {"id": "delta2", "label": "D2", "type": "code", "source_file": "f4.py"},
            {"id": "delta3", "label": "D3", "type": "code", "source_file": "f5.py"},
        ],
        "edges": [
            {"source": "delta1", "target": "old_a", "relation": "calls"},
            {"source": "delta2", "target": "old_b", "relation": "calls"},
            {"source": "delta3", "target": "old_d", "relation": "calls"},
        ],
    }
    ingest_extraction(conn, delta, incremental=True)

    # Analyze with real Leiden -- no mock
    result = analyze_wiki_impact(conn)
    concept_changes = result["concept_changes"]

    # Find which new community contains old_a
    new_communities = result["new_communities"]
    comm_with_old_a = None
    for cid, members in new_communities.items():
        if "old_a" in members:
            comm_with_old_a = cid
            break
    assert comm_with_old_a is not None

    # old_a, old_b, drift, delta1, delta2 should all be in the same community
    comm_members = set(new_communities[comm_with_old_a])
    assert {"old_a", "old_b", "drift", "delta1", "delta2"}.issubset(comm_members)

    # c1 and c2 both dominant to same community -> merge
    c1_change = concept_changes["c1"]
    c2_change = concept_changes["c2"]
    assert c1_change["type"] == "merge"
    assert c2_change["type"] == "merge"
    assert c1_change["target_community"] == comm_with_old_a
    assert "c2" in c1_change["merged_with"]

    # c4 has delta3 -> growth
    c4_change = concept_changes["c4"]
    assert c4_change["type"] == "growth"
    assert "delta3" in c4_change["delta_members"]

    # c1 and c2 co-occur in the same community -> at least a weak link
    all_links = result["link_changes"]["new_links"] + result["link_changes"]["weak_new_links"]
    link_pairs = {(l["from"], l["to"]) for l in all_links}
    assert ("c1", "c2") in link_pairs or ("c2", "c1") in link_pairs

    assert result["concept_names"]["c2"] == "Concept Two"
    _close(db, conn)


def test_format_wiki_impact_shows_delta_and_omits_weak_links():
    from graphify.storage import _format_wiki_impact_text

    text = _format_wiki_impact_text({
        "summary": {
            "stable": 0,
            "growth": 1,
            "merge": 0,
            "split": 0,
            "dissolved": 0,
            "new": 0,
            "new_links": 0,
            "weak_new_links": 1,
            "stale_links": 0,
        },
        "concept_names": {"c1": "Concept One", "c2": "Concept Two"},
        "concept_changes": {
            "c1": {
                "type": "growth",
                "name": "Concept One",
                "old_members": ["old_a", "old_b"],
                "new_members": ["old_a", "old_b", "drift", "delta1", "delta2"],
                "delta_members": ["delta1", "delta2"],
                "old_drift_members": ["drift"],
            }
        },
        "new_concept_candidates": [],
        "link_changes": {
            "new_links": [],
            "weak_new_links": [{"from": "c1", "to": "c2", "co_occurrence": 1}],
            "stale_links": [],
        },
    })

    assert "Concept One: 2 -> 5 members (+2 delta, +1 old-drift)" in text
    assert "weak links:    1 (co-occurrence=1, hidden)" in text
    assert "weak new links omitted (1, co-occurrence=1)" in text
    assert "Concept One -> Concept Two" not in text


def test_run_leiden_warm_start_no_temporary_assignment(tmp_db):
    """Warm-start Leiden on real DB: all nodes assigned, leiden_comm unchanged with write_back=False."""
    from graphify.storage import run_leiden, ingest_extraction
    db, conn = _init(tmp_db)

    # Two disconnected components → Leiden produces 2 communities
    data = {
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
    ingest_extraction(conn, data, incremental=False)

    # Set leiden_comm to known values
    conn.execute("MATCH (n:node) WHERE n.id IN ['n1','n2'] SET n.leiden_comm = 10")
    conn.execute("MATCH (n:node) WHERE n.id IN ['n3','n4'] SET n.leiden_comm = 20")

    # Run with write_back=False — leiden_comm should NOT change
    result = run_leiden(conn, incremental=True, write_back=False)
    # return_previous=False → result is dict[int, list[str]]
    communities: dict = result  # type: ignore[assignment]

    # All nodes must be assigned to a community
    all_assigned = set()
    for members in communities.values():
        all_assigned.update(members)
    assert all_assigned == {"n1", "n2", "n3", "n4"}

    # leiden_comm in DB should be unchanged (write_back=False)
    rows = list(conn.execute("MATCH (n:node) RETURN n.id, n.leiden_comm ORDER BY n.id"))
    comm_map = {r[0]: r[1] for r in rows}
    assert comm_map == {"n1": 10, "n2": 10, "n3": 20, "n4": 20}
    _close(db, conn)


