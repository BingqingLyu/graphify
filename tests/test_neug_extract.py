"""Tests for the NeuG no-graph extract path: to_json_from_extraction (no neug
required) and the cypher analysis helpers cohesion_cypher /
surprising_connections_cypher (neug required)."""
import json
from pathlib import Path

import pytest

try:
    import neug  # noqa: F401
    _has_neug = True
except ImportError:
    _has_neug = False


# --- to_json_from_extraction (pure Python, no neug) ---

def test_to_json_from_extraction_format(tmp_path):
    from graphify.export import to_json_from_extraction
    from graphify.build import build_from_json

    merged = {
        "nodes": [
            {"id": "a", "label": "A", "type": "code", "source_file": "x.py"},
            {"id": "b", "label": "B", "type": "code", "source_file": "x.py"},
            {"id": "c", "label": "C", "type": "code", "source_file": "y.py"},
        ],
        "edges": [
            {"source": "a", "target": "b", "relation": "calls", "confidence": "EXTRACTED"},
            {"source": "b", "target": "c", "relation": "uses", "confidence": "INFERRED"},
        ],
    }
    communities = {0: ["a", "b"], 1: ["c"]}
    out = tmp_path / "graph.json"

    ok = to_json_from_extraction(
        merged, communities, str(out), force=True, built_at_commit="deadbeef"
    )
    assert ok

    data = json.loads(out.read_text())
    # node-link shape, matching to_json / build()'s directed=False default
    assert data["directed"] is False
    assert data["multigraph"] is False
    assert len(data["nodes"]) == 3
    assert len(data["links"]) == 2
    assert data["built_at_commit"] == "deadbeef"

    by_id = {n["id"]: n for n in data["nodes"]}
    assert by_id["a"]["community"] == 0
    assert by_id["c"]["community"] == 1
    assert "norm_label" in by_id["a"]

    for link in data["links"]:
        assert "confidence_score" in link
        assert "source" in link and "target" in link

    # INFERRED default confidence_score
    inferred = [l for l in data["links"] if l["confidence"] == "INFERRED"][0]
    assert inferred["confidence_score"] == 0.5

    # downstream reader works unchanged (build_from_json takes a dict; the
    # links->edges remap is exercised here)
    G = build_from_json(data)
    assert G.number_of_nodes() == 3
    assert G.number_of_edges() == 2


def test_to_json_from_extraction_dedupes(tmp_path):
    from graphify.export import to_json_from_extraction

    merged = {
        "nodes": [
            {"id": "a", "label": "A", "type": "code", "source_file": "x.py"},
            {"id": "a", "label": "A2", "type": "code", "source_file": "x.py"},
            {"id": "b", "label": "B", "type": "code", "source_file": "x.py"},
        ],
        "edges": [
            {"source": "a", "target": "b", "relation": "calls", "confidence": "EXTRACTED"},
            {"source": "a", "target": "b", "relation": "calls", "confidence": "EXTRACTED"},
        ],
    }
    out = tmp_path / "graph.json"
    to_json_from_extraction(merged, {0: ["a", "b"]}, str(out), force=True)
    data = json.loads(out.read_text())
    assert len(data["nodes"]) == 2  # same-id collapsed
    assert len(data["links"]) == 1  # parallel edge collapsed


# --- cypher analysis helpers (neug required) ---

@pytest.mark.skipif(not _has_neug, reason="neug not installed")
def test_cohesion_cypher(tmp_path):
    from graphify.storage import (
        init_db, ensure_schema, ingest_extraction, close_db, cohesion_cypher,
    )
    db, conn = init_db(str(tmp_path / "t.db"))
    ensure_schema(conn)
    ext = {
        "nodes": [
            {"id": "a", "label": "A", "type": "code", "source_file": "x.py"},
            {"id": "b", "label": "B", "type": "code", "source_file": "x.py"},
            {"id": "c", "label": "C", "type": "code", "source_file": "x.py"},
            {"id": "d", "label": "D", "type": "code", "source_file": "y.py"},
        ],
        "edges": [
            {"source": "a", "target": "b", "relation": "calls", "confidence": "EXTRACTED"},
            {"source": "b", "target": "c", "relation": "calls", "confidence": "EXTRACTED"},
            {"source": "a", "target": "c", "relation": "calls", "confidence": "EXTRACTED"},
        ],
    }
    ingest_extraction(conn, ext, incremental=False)
    coh = cohesion_cypher(conn, {0: ["a", "b", "c"], 1: ["d"]})
    # comm0: 3 nodes, 3 intra edges, possible = 3 -> 1.0
    assert coh[0] == pytest.approx(1.0)
    # comm1: single node -> 1.0
    assert coh[1] == pytest.approx(1.0)
    close_db(db, conn)


@pytest.mark.skipif(not _has_neug, reason="neug not installed")
def test_cohesion_cypher_partial(tmp_path):
    from graphify.storage import (
        init_db, ensure_schema, ingest_extraction, close_db, cohesion_cypher,
    )
    db, conn = init_db(str(tmp_path / "t.db"))
    ensure_schema(conn)
    ext = {
        "nodes": [
            {"id": "a", "label": "A", "type": "code", "source_file": "x.py"},
            {"id": "b", "label": "B", "type": "code", "source_file": "x.py"},
            {"id": "c", "label": "C", "type": "code", "source_file": "x.py"},
        ],
        "edges": [
            {"source": "a", "target": "b", "relation": "calls", "confidence": "EXTRACTED"},
        ],
    }
    ingest_extraction(conn, ext, incremental=False)
    coh = cohesion_cypher(conn, {0: ["a", "b", "c"]})
    # 3 nodes, 1 intra edge, possible = 3 -> 1/3
    assert coh[0] == pytest.approx(1.0 / 3.0)
    close_db(db, conn)


@pytest.mark.skipif(not _has_neug, reason="neug not installed")
def test_surprising_connections_cypher(tmp_path):
    from graphify.storage import (
        init_db, ensure_schema, ingest_extraction, close_db,
        surprising_connections_cypher,
    )
    db, conn = init_db(str(tmp_path / "t.db"))
    ensure_schema(conn)
    ext = {
        "nodes": [
            {"id": "a", "label": "funcA", "type": "code", "source_file": "pkg/x.py"},
            {"id": "b", "label": "funcB", "type": "code", "source_file": "pkg/x.py"},
            {"id": "c", "label": "funcC", "type": "code", "source_file": "other/y.py"},
            {"id": "h", "label": "hub", "type": "code", "source_file": "pkg/x.py"},
        ],
        "edges": [
            {"source": "a", "target": "b", "relation": "calls", "confidence": "EXTRACTED"},
            # cross-file + cross-dir + cross-community EXTRACTED edge -> surprising
            {"source": "a", "target": "c", "relation": "calls", "confidence": "EXTRACTED"},
            {"source": "h", "target": "a", "relation": "calls", "confidence": "EXTRACTED"},
            {"source": "h", "target": "b", "relation": "calls", "confidence": "EXTRACTED"},
            # structural edge must be excluded
            {"source": "h", "target": "c", "relation": "contains", "confidence": "EXTRACTED"},
        ],
    }
    ingest_extraction(conn, ext, incremental=False)
    sur = surprising_connections_cypher(conn, {0: ["a", "b", "h"], 1: ["c"]}, top_n=5)
    assert isinstance(sur, list)
    pairs = {(s["source"], s["target"]) for s in sur}
    assert ("funcA", "funcC") in pairs
    # structural contains edge never surfaces
    assert ("hub", "funcC") not in pairs
    for s in sur:
        assert len(s["source_files"]) == 2
        assert "relation" in s and "confidence" in s
    close_db(db, conn)


@pytest.mark.skipif(not _has_neug, reason="neug not installed")
def test_surprising_connections_cypher_empty(tmp_path):
    from graphify.storage import (
        init_db, ensure_schema, ingest_extraction, close_db,
        surprising_connections_cypher,
    )
    db, conn = init_db(str(tmp_path / "t.db"))
    ensure_schema(conn)
    ext = {
        "nodes": [
            {"id": "a", "label": "A", "type": "code", "source_file": "x.py"},
            {"id": "b", "label": "B", "type": "code", "source_file": "x.py"},
        ],
        # only same-file structural-ish edge -> no cross-file surprises
        "edges": [
            {"source": "a", "target": "b", "relation": "contains", "confidence": "EXTRACTED"},
        ],
    }
    ingest_extraction(conn, ext, incremental=False)
    sur = surprising_connections_cypher(conn, {0: ["a", "b"]})
    assert sur == []
    close_db(db, conn)
