"""NeuG-optimized extract pipeline (graph.db native, no NetworkX graph).

This module is the neug-backed counterpart to the extract full/incremental path
in ``__main__.py``. It is intentionally kept **completely separate** from the
original NetworkX pipeline: ``__main__`` calls :func:`neug_extract` only when the
``neug`` package is importable, and otherwise falls back to the untouched
NetworkX path. The two never mix.

Why: for large repos the NetworkX path holds the AST residue + ``merged`` dict +
a full in-memory graph + the NeuG Leiden projection simultaneously, and the
memory peak causes OOM/segfault. Here we ingest straight into graph.db, run
Leiden on the connection, and compute all analysis (gods/cohesion/surprises)
with cypher — never materializing a NetworkX graph.

Scope (this iteration): the full and incremental extract flows. ``--no-cluster``
already runs a lightweight raw-dump + ``neug_sync`` (no graph is built there), so
it keeps its existing path in ``__main__`` and is not routed through here.
"""

from __future__ import annotations

import gc
import json
import sys
from pathlib import Path
from typing import Any


def _leiden_communities(conn: Any, resolution: float, incremental: bool) -> dict[int, list[str]]:
    """Run NeuG native Leiden and re-index communities by size.

    Mirrors the NeuG branch of ``cluster.cluster`` (deterministic re-index,
    skipping the NetworkX-dependent postprocess) so IDs are stable without a
    graph in memory.
    """
    from graphify.storage import run_leiden

    neu_raw = run_leiden(conn, resolution=resolution, incremental=incremental)
    if not neu_raw:
        return {}
    communities_sorted = sorted(
        neu_raw.values(),
        key=lambda nodes: (-len(nodes), tuple(sorted(map(str, nodes)))),
    )
    return {i: sorted(nodes) for i, nodes in enumerate(communities_sorted)}


def _count(conn: Any, query: str, fallback: int) -> int:
    try:
        rows = list(conn.execute(query))
        return int(rows[0][0]) if rows and rows[0][0] is not None else fallback
    except Exception:
        return fallback


def neug_extract(
    merged: dict,
    *,
    graphify_out: Path,
    target: Path,
    graph_json_path: Path,
    analysis_path: Path,
    manifest_path: Path,
    manifest_files: dict,
    incremental_mode: bool = False,
    deleted_files: list[str] | None = None,
    resolution: float = 1.0,
    backend: str = "",
    global_merge: bool = False,
    global_repo_tag: str | None = None,
    sem_cache_hits: int = 0,
    sem_cache_misses: int = 0,
    unchanged_total: int = 0,
    code_files: list | None = None,
    stages: Any = None,
) -> None:
    """Run the full/incremental extract pipeline against graph.db (no NetworkX).

    Produces the same artefacts as the NetworkX path — graph.json (node-link),
    ``.graphify_analysis.json`` (communities/cohesion/gods/surprises), graph.db
    concepts, manifest — so every downstream reader keeps working unchanged.

    Raises to let ``__main__`` fall back to the NetworkX path if the neug write
    pipeline fails midway (``SystemExit`` from the empty-graph guard propagates).
    """
    from graphify.storage import (
        neug_sync,
        close_db,
        ingest_concepts,
        cohesion_cypher,
        surprising_connections_cypher,
        god_nodes_cypher,
    )
    from graphify.export import (
        to_json_from_extraction,
        backup_if_protected,
    )
    from graphify.detect import save_manifest
    from graphify.llm import estimate_cost

    def _mark(name: str) -> None:
        if stages is not None:
            try:
                stages.mark(name)
            except Exception:
                pass

    # Empty-graph guard (parity with the NetworkX path's number_of_nodes()==0).
    if not merged.get("nodes"):
        print(
            "[graphify extract] graph is empty — extraction produced no nodes. "
            "Possible causes: all files skipped, binary-only corpus, or LLM "
            "returned no edges.",
            file=sys.stderr,
        )
        sys.exit(1)

    # 1. Ingest extraction into graph.db and keep the connection open.
    db_path = str(graphify_out / "graph.db")
    handle = neug_sync(
        db_path, merged,
        incremental=Path(db_path).exists(),
        prune_sources=deleted_files or None, root=target,
    )
    if handle is None:
        raise RuntimeError("neug_sync returned no connection")
    db_, conn = handle
    _mark("ingest")

    try:
        # 2. Cluster with NeuG Leiden directly on the connection (no graph).
        communities = _leiden_communities(conn, resolution, incremental_mode)
        _mark("cluster")

        # 3. graph.json straight from merged + communities (no graph).
        backup_if_protected(graphify_out)
        to_json_from_extraction(merged, communities, str(graph_json_path), force=True)
        _mark("export")

        # Capture token/size stats before releasing merged to trim memory peak.
        in_tokens = int(merged.get("input_tokens", 0) or 0)
        out_tokens = int(merged.get("output_tokens", 0) or 0)
        n_nodes_fallback = len(merged.get("nodes", []))
        n_edges_fallback = len(merged.get("edges", []))
        del merged
        gc.collect()

        # 4. Analysis entirely via cypher — no NetworkX graph.
        cohesion = cohesion_cypher(conn, communities)
        try:
            gods = god_nodes_cypher(conn)
        except Exception:
            gods = []
        try:
            surprises = surprising_connections_cypher(conn, communities)
        except Exception:
            surprises = []
        _mark("analyze")

        n_nodes = _count(conn, "MATCH (n:node) RETURN count(n)", n_nodes_fallback)
        n_edges = _count(conn, "MATCH ()-[e:edge]->() RETURN count(e)", n_edges_fallback)

        # 5. Persist clustering results (concepts) to graph.db.
        try:
            ingest_concepts(conn, [
                {"id": f"concept_{cid}", "name": f"Community {cid}",
                 "source": "leiden", "members": members}
                for cid, members in communities.items()
            ])
        except Exception as exc:
            print(f"[graphify extract] warning: NeuG concept write failed: {exc}", file=sys.stderr)
    finally:
        close_db(db_, conn)
    print("[graphify extract] graph.db written (powered by NeuG)")

    # 6. Semantic marker (mirrors the NetworkX path).
    if out_tokens > 0:
        (graphify_out / ".graphify_semantic_marker").write_text(
            json.dumps({"output_tokens": out_tokens}), encoding="utf-8"
        )

    # 7. Optional global-graph merge.
    if global_merge:
        from graphify.global_graph import global_add as _global_add
        _tag = global_repo_tag or target.name
        try:
            result = _global_add(graphify_out / "graph.json", _tag)
            if result["skipped"]:
                print(f"[graphify global] '{_tag}' unchanged since last add - skipped.")
            else:
                print(f"[graphify global] '{_tag}' merged into global graph "
                      f"(+{result['nodes_added']} nodes, -{result['nodes_removed']} pruned).")
        except Exception as exc:
            print(f"[graphify global] warning: failed to merge into global graph: {exc}", file=sys.stderr)

    # 8. Analysis sidecar (identical schema to the NetworkX path).
    analysis = {
        "communities": {str(k): v for k, v in communities.items()},
        "cohesion": {str(k): v for k, v in cohesion.items()},
        "gods": gods,
        "surprises": surprises,
        "tokens": {"input": in_tokens, "output": out_tokens},
    }
    analysis_path.write_text(json.dumps(analysis, indent=2), encoding="utf-8")

    try:
        save_manifest(manifest_files, manifest_path=str(manifest_path), kind="both", root=target)
    except Exception as exc:
        print(f"[graphify extract] warning: could not write manifest: {exc}", file=sys.stderr)

    cost = estimate_cost(backend, in_tokens, out_tokens)
    print(
        f"[graphify extract] wrote {graph_json_path}: "
        f"{n_nodes} nodes, {n_edges} edges, {len(communities)} communities"
    )
    print(f"[graphify extract] wrote {analysis_path}")
    if incremental_mode:
        _deleted_n = len(deleted_files) if deleted_files else 0
        _code_n = len(code_files) if code_files else 0
        print(
            f"[graphify extract] incremental summary: "
            f"{sem_cache_hits + unchanged_total} files cached/unchanged, "
            f"{_code_n + sem_cache_misses} re-extracted, "
            f"{_deleted_n} deleted"
        )
    elif sem_cache_hits:
        print(f"[graphify extract] semantic cache: {sem_cache_hits} cached, {sem_cache_misses} re-extracted")
    if in_tokens or out_tokens:
        print(
            f"[graphify extract] tokens: "
            f"{in_tokens:,} in / {out_tokens:,} out, "
            f"est. cost (~{backend}): ${cost:.4f}"
        )
    print(
        "[graphify extract] next: run "
        f"`graphify cluster-only {graphify_out.parent}` "
        "to generate GRAPH_REPORT.md and name communities"
    )
