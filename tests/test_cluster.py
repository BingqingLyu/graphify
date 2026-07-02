import json
import sys
import networkx as nx
from pathlib import Path
from graphify.build import build_from_json
from graphify.cluster import (
    cluster,
    cohesion_score,
    postprocess_communities,
    _extract_and_reattach_hubs,
    remap_communities_to_previous,
    score_all,
)

FIXTURES = Path(__file__).parent / "fixtures"

def make_graph():
    return build_from_json(json.loads((FIXTURES / "extraction.json").read_text()))

def test_cluster_returns_dict():
    G = make_graph()
    communities = cluster(G)
    assert isinstance(communities, dict)

def test_cluster_covers_all_nodes():
    G = make_graph()
    communities = cluster(G)
    all_nodes = {n for nodes in communities.values() for n in nodes}
    assert all_nodes == set(G.nodes)

def test_cohesion_score_complete_graph():
    G = nx.complete_graph(4)
    G = nx.relabel_nodes(G, {i: str(i) for i in G.nodes})
    score = cohesion_score(G, list(G.nodes))
    assert score == 1.0

def test_cohesion_score_single_node():
    G = nx.Graph()
    G.add_node("a")
    score = cohesion_score(G, ["a"])
    assert score == 1.0

def test_cohesion_score_disconnected():
    G = nx.Graph()
    G.add_nodes_from(["a", "b", "c"])
    score = cohesion_score(G, ["a", "b", "c"])
    assert score == 0.0

def test_cohesion_score_range():
    G = make_graph()
    communities = cluster(G)
    for cid, nodes in communities.items():
        score = cohesion_score(G, nodes)
        assert 0.0 <= score <= 1.0

def test_score_all_keys_match_communities():
    G = make_graph()
    communities = cluster(G)
    scores = score_all(G, communities)
    assert set(scores.keys()) == set(communities.keys())


def test_cluster_does_not_write_to_stdout(capsys):
    """Clustering should not emit ANSI escape codes or other output.

    graspologic's leiden() can emit ANSI escape sequences that break
    PowerShell 5.1's scroll buffer on Windows (issue #19). The output
    suppression in _partition() should prevent any output from leaking.
    """
    G = make_graph()
    cluster(G)
    captured = capsys.readouterr()
    assert captured.out == "", f"cluster() wrote to stdout: {captured.out!r}"


def test_cluster_does_not_write_to_stderr(capsys):
    """Same as above but for stderr — ANSI codes can go to either stream."""
    G = make_graph()
    cluster(G)
    captured = capsys.readouterr()
    # Allow logging output (starts with [graphify]) but no raw ANSI codes
    for line in captured.err.splitlines():
        assert "\x1b" not in line, f"cluster() wrote ANSI to stderr: {line!r}"


def test_remap_communities_to_previous_reuses_old_ids():
    communities = {
        10: ["a", "b", "c"],
        11: ["d", "e"],
    }
    previous = {"a": 5, "b": 5, "c": 5, "d": 1, "e": 1}
    remapped = remap_communities_to_previous(communities, previous)
    assert set(remapped.keys()) == {1, 5}
    assert remapped[5] == ["a", "b", "c"]
    assert remapped[1] == ["d", "e"]


def test_remap_communities_to_previous_assigns_deterministic_new_ids():
    communities = {
        7: ["x", "y", "z"],
        8: ["m"],
    }
    previous = {"a": 3}
    remapped = remap_communities_to_previous(communities, previous)
    assert list(remapped.keys()) == [0, 1]
    assert remapped[0] == ["x", "y", "z"]
    assert remapped[1] == ["m"]


# --- postprocess_communities ---

def test_postprocess_communities_reindexes_by_size_desc():
    """Largest community should get ID 0 after postprocessing."""
    G = nx.Graph()
    # Add edges within each group so they survive the oversized-split check
    for i in range(15):
        for j in range(i + 1, 15):
            G.add_edge(f"n{i}", f"n{j}")
    for i in range(15, 20):
        for j in range(i + 1, 20):
            G.add_edge(f"n{i}", f"n{j}")
    raw = {
        100: [f"n{i}" for i in range(15)],  # largest
        200: [f"n{i}" for i in range(15, 20)],  # smaller
    }
    result = postprocess_communities(G, raw)
    # ID 0 = largest, ID 1 = smaller
    assert len(result[0]) == 15
    assert len(result[1]) == 5


def test_postprocess_communities_splits_oversized():
    """Community >25% of graph nodes should be split."""
    G = nx.Graph()
    # Create a graph with two dense clusters connected by a bridge
    for i in range(20):
        for j in range(i + 1, 20):
            G.add_edge(f"a{i}", f"a{j}")
    for i in range(20):
        for j in range(i + 1, 20):
            G.add_edge(f"b{i}", f"b{j}")
    G.add_edge("a0", "b0")  # bridge

    raw = {0: [f"a{i}" for i in range(20)] + [f"b{i}" for i in range(20)]}
    result = postprocess_communities(G, raw)
    # The oversized community should be split into at least 2
    assert len(result) >= 2


def test_postprocess_communities_empty():
    """Empty input should return empty dict."""
    G = nx.Graph()
    assert postprocess_communities(G, {}) == {}


# --- _extract_and_reattach_hubs ---

def test_extract_and_reattach_hubs_majority_vote():
    """Hub nodes should be reattached to their majority-vote neighbour community."""
    G = nx.Graph()
    # Community A: 4 nodes, Community B: 2 nodes, Hub connected to both
    for i in range(4):
        G.add_node(f"a{i}")
    for i in range(2):
        G.add_node(f"b{i}")
    G.add_node("hub")
    # Hub connects to 3 A-nodes and 1 B-node
    G.add_edges_from([("hub", "a0"), ("hub", "a1"), ("hub", "a2"), ("hub", "b0")])

    raw = {0: ["a0", "a1", "a2", "a3", "hub"], 1: ["b0", "b1"]}
    _extract_and_reattach_hubs(G, raw, {"hub"})

    # Hub should be in community 0 (3 A-neighbours vs 1 B-neighbour)
    assert "hub" in raw[0]
    assert "hub" not in raw.get(1, [])


def test_extract_and_reattach_hubs_isolated_hub():
    """Hub with no neighbours in any community gets its own community."""
    G = nx.Graph()
    G.add_nodes_from(["a", "hub"])
    raw = {0: ["a", "hub"]}
    _extract_and_reattach_hubs(G, raw, {"hub"})
    # Hub removed from community 0
    assert "hub" not in raw[0]
    # Hub placed in its own community
    hub_communities = [c for c in raw.values() if "hub" in c]
    assert len(hub_communities) == 1


# --- cluster with conn parameter ---

def test_cluster_with_none_conn_uses_python_path():
    """cluster() with conn=None should use the Python partitioning path."""
    G = make_graph()
    communities = cluster(G, conn=None)
    assert isinstance(communities, dict)
    all_nodes = {n for nodes in communities.values() for n in nodes}
    assert all_nodes == set(G.nodes)


def test_cluster_with_failing_conn_falls_back():
    """cluster() should fall back to Python path when NeuG Leiden fails."""
    G = make_graph()

    class FailingConn:
        def execute(self, *args, **kwargs):
            raise RuntimeError("NeuG connection failed")

    communities = cluster(G, conn=FailingConn())
    assert isinstance(communities, dict)
    all_nodes = {n for nodes in communities.values() for n in nodes}
    assert all_nodes == set(G.nodes)


def test_cluster_with_empty_neug_result_falls_back():
    """cluster() should fall back to Python when NeuG returns empty results."""
    G = make_graph()

    class EmptyConn:
        def execute(self, *args, **kwargs):
            return iter([])

    communities = cluster(G, conn=EmptyConn())
    assert isinstance(communities, dict)
    all_nodes = {n for nodes in communities.values() for n in nodes}
    assert all_nodes == set(G.nodes)
