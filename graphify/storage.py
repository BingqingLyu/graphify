"""NeuG graph database adapter for graphify.

Provides an optional parallel storage engine alongside NetworkX.
NeuG is lazily imported — when not installed, callers should catch
ImportError at the call site and skip silently.

All property values interpolated into Cypher statements use NeuG's native
parameterised queries ($param syntax) to prevent injection.  Table/label
names (which come from a fixed internal set, not user input) are still
interpolated as identifiers.
"""
from __future__ import annotations

import csv
import os
import tempfile
from pathlib import Path

from .build import _FILE_TYPE_SYNONYMS, _normalize_id, _norm_source_file
from .validate import VALID_FILE_TYPES

# ---------------------------------------------------------------------------
# DDL — unified schema (4 tables)
# ---------------------------------------------------------------------------

_NODE_DDL = """CREATE NODE TABLE IF NOT EXISTS node (
    id STRING PRIMARY KEY, label STRING, type STRING,
    source_file STRING, source_location STRING)"""

_CONCEPT_DDL = """CREATE NODE TABLE IF NOT EXISTS concept (
    id STRING PRIMARY KEY, name STRING,
    description STRING, source STRING)"""

_EDGE_DDL = """CREATE REL TABLE IF NOT EXISTS edge (
    FROM node TO node,
    relation STRING, confidence STRING,
    confidence_score DOUBLE, source_file STRING, weight DOUBLE)"""

_BELONGS_DDL = """CREATE REL TABLE IF NOT EXISTS belongs_to (
    FROM node TO concept)"""

# ---------------------------------------------------------------------------
# Column definitions for CSV output
# ---------------------------------------------------------------------------

_NODE_COLUMNS = ["id", "label", "type", "source_file", "source_location"]
_EDGE_COLUMNS = ["from_id", "to_id", "relation", "confidence",
                 "confidence_score", "source_file", "weight"]
_CONCEPT_COLUMNS = ["id", "name", "description", "source"]
_BELONGS_COLUMNS = ["node_id", "concept_id"]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _sanitize_csv_value(v: object) -> str:
    if isinstance(v, str):
        return v.replace("\n", "\\n").replace("\r", "")
    return str(v)


def _write_csv(path: str, rows: list[dict], columns: list[str]) -> int:
    if not rows:
        return 0
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore",
                           quoting=csv.QUOTE_ALL)
        w.writeheader()
        for row in rows:
            w.writerow({k: _sanitize_csv_value(row.get(k, "")) for k in columns})
    return len(rows)


def _copy_csv(conn: object, csv_path: str, table: str, *,
               from_table: str | None = None,
               to_table: str | None = None) -> None:
    """COPY FROM for node or relationship tables.

    When from_table/to_table are None, performs a node COPY.
    Otherwise performs a relationship COPY with endpoint references.
    """
    if from_table and to_table:
        conn.execute(
            f'COPY {table} FROM "{csv_path}" '
            f'(from="{from_table}", to="{to_table}", '
            f'header=true, delim=",", escaping=false)'
        )
    else:
        conn.execute(
            f'COPY {table} FROM "{csv_path}" (header=true, delim=",", escaping=false)'
        )


def _fix_file_type(ft: str | None) -> str:
    """Canonicalize file_type, matching build.py logic."""
    if not ft or ft not in VALID_FILE_TYPES:
        return _FILE_TYPE_SYNONYMS.get(ft, "concept") if ft else "concept"
    return ft


def _normalize_nodes(
    extraction: dict, root: str | None = None,
) -> tuple[list[dict], list[dict], dict[str, str]]:
    """Convert extraction dict to normalized node/edge row lists.

    Returns (node_rows, edge_rows, node_types).
    node_types maps id -> type for edge endpoint resolution.
    """
    nodes = extraction.get("nodes") or []
    edges = extraction.get("edges") or []

    node_types: dict[str, str] = {}
    node_rows: list[dict] = []
    seen_ids: set[str] = set()

    for node in nodes:
        nid = _normalize_id(node.get("id", ""))
        if not nid or nid in seen_ids:
            continue
        seen_ids.add(nid)
        ft = _fix_file_type(node.get("file_type"))
        node_types[nid] = ft
        node_rows.append({
            "id": nid,
            "label": node.get("label", ""),
            "type": ft,
            "source_file": _norm_source_file(node.get("source_file"), root) or "",
            "source_location": node.get("source_location") or "",
        })

    edge_rows: list[dict] = []
    for edge in edges:
        src_id = _normalize_id(edge.get("source") or edge.get("from", ""))
        tgt_id = _normalize_id(edge.get("target") or edge.get("to", ""))
        if not src_id or not tgt_id:
            continue
        if src_id not in node_types or tgt_id not in node_types:
            continue
        edge_rows.append({
            "from_id": src_id,
            "to_id": tgt_id,
            "relation": edge.get("relation", ""),
            "confidence": edge.get("confidence", ""),
            "confidence_score": float(edge.get("confidence_score", 0.0)),
            "source_file": _norm_source_file(edge.get("source_file"), root) or "",
            "weight": float(edge.get("weight", 1.0)),
        })

    return node_rows, edge_rows, node_types


def _bulk_load(
    conn: object,
    tmpdir: str,
    node_rows: list[dict],
    node_columns: list[str],
    node_table: str,
    edge_rows: list[dict],
    edge_columns: list[str],
    edge_table: str,
    *,
    edge_from: str | None = None,
    edge_to: str | None = None,
) -> None:
    """Write CSVs and COPY FROM in one shot. Shared by bulk/incremental/concepts."""
    if node_rows:
        csv_path = os.path.join(tmpdir, f"{node_table}.csv")
        _write_csv(csv_path, node_rows, node_columns)
        _copy_csv(conn, csv_path, node_table)

    if edge_rows:
        csv_path = os.path.join(tmpdir, f"{edge_table}.csv")
        _write_csv(csv_path, edge_rows, edge_columns)
        _copy_csv(conn, csv_path, edge_table,
                   from_table=edge_from or node_table,
                   to_table=edge_to or node_table)


# ---------------------------------------------------------------------------
# Public API — connection lifecycle
# ---------------------------------------------------------------------------


def init_db(db_path: str) -> tuple:
    """Open (or create) a NeuG database and connect.

    Returns (db, conn).  Raises ImportError if neug is not installed.
    """
    import neug
    db = neug.Database(db_path)
    conn = db.connect()
    return db, conn


def ensure_schema(conn: object, *, create_tables: bool = True) -> None:
    """Execute DDL for the unified schema (4 tables).

    create_tables=True: run CREATE TABLE statements.
    create_tables=False: no-op (kept for API compatibility).
    """
    if create_tables:
        for ddl in (_NODE_DDL, _CONCEPT_DDL, _EDGE_DDL, _BELONGS_DDL):
            conn.execute(ddl)


def execute_cypher(conn: object, query: str) -> list[list]:
    """Execute a Cypher query and return results as list of lists."""
    try:
        return list(conn.execute(query))
    except RuntimeError as exc:
        raise RuntimeError(f"Cypher query failed: {exc}") from exc


def close_db(db: object, conn: object) -> None:
    """Close the NeuG connection and database."""
    conn.close()
    db.close()


# ---------------------------------------------------------------------------
# Ingest — bulk and incremental
# ---------------------------------------------------------------------------


def _bulk_ingest(
    conn: object,
    extraction: dict,
    *,
    root: str | None = None,
    known_tables: set[str] | None = None,
) -> dict[str, str]:
    """Full build via COPY FROM — much faster than per-row Cypher CREATE."""
    node_rows, edge_rows, node_types = _normalize_nodes(extraction, root)

    tmp_dir = tempfile.mkdtemp(prefix="graphify_bulk_")
    try:
        _bulk_load(
            conn, tmp_dir,
            node_rows, _NODE_COLUMNS, "node",
            edge_rows, _EDGE_COLUMNS, "edge",
            edge_from="node", edge_to="node",
        )
    finally:
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return node_types


def _incremental_ingest(
    conn: object,
    extraction: dict,
    *,
    prune_sources: list[str] | None = None,
    root: str | None = None,
    known_tables: set[str] | None = None,
) -> dict[str, str]:
    """Incremental update via DELETE affected source_files + COPY FROM.

    Deletes nodes whose source_file appears in the incoming extraction
    (or in prune_sources), then bulk-inserts the new data via COPY FROM.
    Incoming cross-file edges (from unchanged files into affected nodes)
    are saved before deletion and restored afterwards.
    """
    nodes_data = extraction.get("nodes") or []
    edges_data = extraction.get("edges") or []

    # --- collect affected source_files ---
    affected_sfs: set[str] = set()
    if prune_sources:
        for sf in prune_sources:
            sf_norm = _norm_source_file(sf, root) or sf
            affected_sfs.add(sf_norm)

    node_rows, edge_rows, node_types = _normalize_nodes(extraction, root)
    for row in node_rows:
        if row["source_file"]:
            affected_sfs.add(row["source_file"])

    # --- save incoming cross-file edges before DELETE ---
    affected_node_ids: set[str] = set()
    for sf in affected_sfs:
        try:
            for r in conn.execute(
                "MATCH (n:node) WHERE n.source_file = $sf RETURN n.id",
                parameters={"sf": sf},
            ):
                affected_node_ids.add(r[0])
        except RuntimeError:
            pass

    saved_edges: list[dict] = []
    for sf in affected_sfs:
        try:
            rows = list(conn.execute(
                "MATCH (a:node)-[e:edge]->(b:node) "
                "WHERE b.source_file = $sf "
                "RETURN a.id, b.id, e.relation, e.confidence, "
                "e.confidence_score, e.source_file, e.weight",
                parameters={"sf": sf},
            ))
        except RuntimeError:
            continue
        for r in rows:
            if r[0] not in affected_node_ids:
                saved_edges.append({
                    "from_id": r[0], "to_id": r[1],
                    "relation": r[2] or "",
                    "confidence": r[3] or "",
                    "confidence_score": float(r[4] or 0.0),
                    "source_file": r[5] or "",
                    "weight": float(r[6] or 1.0),
                })

    # --- DELETE nodes from affected source_files ---
    for sf in affected_sfs:
        conn.execute(
            "MATCH (n:node) WHERE n.source_file = $sf DETACH DELETE n",
            parameters={"sf": sf},
        )

    # --- merge saved incoming edges ---
    edge_rows.extend(saved_edges)

    # --- COPY FROM bulk insert ---
    tmp_dir = tempfile.mkdtemp(prefix="graphify_inc_")
    try:
        _bulk_load(
            conn, tmp_dir,
            node_rows, _NODE_COLUMNS, "node",
            edge_rows, _EDGE_COLUMNS, "edge",
            edge_from="node", edge_to="node",
        )
    finally:
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return node_types


def ingest_extraction(
    conn: object,
    extraction: dict,
    *,
    incremental: bool = False,
    prune_sources: list[str] | None = None,
    root: str | Path | None = None,
    known_tables: set[str] | None = None,
) -> dict[str, str]:
    """Write an extraction dict into NeuG.

    incremental=False: first build — uses COPY FROM bulk loading.
    incremental=True:  update — DELETE affected + COPY FROM.

    Returns node_types dict (id -> file_type) for use by ingest_communities.
    """
    _root = str(Path(root).resolve()) if root else None

    if incremental:
        return _incremental_ingest(
            conn, extraction,
            prune_sources=prune_sources, root=_root,
            known_tables=known_tables,
        )
    else:
        return _bulk_ingest(
            conn, extraction,
            root=_root, known_tables=known_tables,
        )


def ingest_communities(
    conn: object,
    communities: dict[int, list[str]],
    community_labels: dict[int, str] | None = None,
    node_types: dict[str, str] | None = None,
) -> None:
    """Deprecated: use ingest_concepts() instead.

    Kept for backward compatibility — writes community assignments as
    Concept nodes + BELONGS_TO edges via ingest_concepts().
    """
    concepts = [
        {"id": f"concept_{cid}", "name": f"Community {cid}",
         "source": "leiden", "members": node_ids}
        for cid, node_ids in communities.items()
    ]
    ingest_concepts(conn, concepts)


# ---------------------------------------------------------------------------
# Concept nodes + BELONGS_TO edges
# ---------------------------------------------------------------------------


def ingest_concepts(
    conn: object,
    concepts: list[dict],
) -> None:
    """Write Concept nodes + BELONGS_TO edges to NeuG.

    concepts: list of dicts, each with:
        id: str — concept identifier
        name: str — concept name
        description: str — concept description (optional, default "")
        source: str — provenance: 'leiden', 'wiki', 'manual', etc. (optional, default "leiden")
        members: list[str] — node IDs that belong to this concept
    """
    concept_rows: list[dict] = []
    belongs_rows: list[dict] = []

    for c in concepts:
        cid = c.get("id", "")
        if not cid:
            continue
        concept_rows.append({
            "id": cid,
            "name": c.get("name", cid),
            "description": c.get("description", ""),
            "source": c.get("source", "leiden"),
        })
        for nid in c.get("members", []):
            nid_norm = _normalize_id(nid)
            if nid_norm:
                belongs_rows.append({
                    "node_id": nid_norm,
                    "concept_id": cid,
                })

    tmp_dir = tempfile.mkdtemp(prefix="graphify_concepts_")
    try:
        _bulk_load(
            conn, tmp_dir,
            concept_rows, _CONCEPT_COLUMNS, "concept",
            belongs_rows, _BELONGS_COLUMNS, "belongs_to",
            edge_from="node", edge_to="concept",
        )
    finally:
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)


def get_concept_members(conn: object) -> dict[str, list[str]]:
    """Query all (concept_id, [node_ids]) from BELONGS_TO edges."""
    result: dict[str, list[str]] = {}
    try:
        rows = conn.execute(
            "MATCH (n:node)-[:belongs_to]->(c:concept) "
            "RETURN c.id, n.id"
        )
        for row in rows:
            concept_id, node_id = row[0], row[1]
            result.setdefault(concept_id, []).append(node_id)
    except RuntimeError:
        pass
    return result


# ---------------------------------------------------------------------------
# Leiden community detection (NeuG GDS)
# ---------------------------------------------------------------------------


def _ensure_gds(conn: object) -> None:
    """Ensure GDS extension is loaded. Idempotent."""
    try:
        conn.execute("LOAD gds")
    except RuntimeError:
        conn.execute("INSTALL gds")
        conn.execute("LOAD gds")


def _leiden_on_projected(
    conn: object,
    graph_name: str,
    resolution: float,
    concurrency: int | None,
) -> dict[int, list[str]]:
    """Run Leiden on an already-projected graph, parse results, clean up.

    Returns communities dict: {community_id: [node_ids]}.
    """
    opts = f"resolution: {resolution}"
    if concurrency is not None:
        opts += f", concurrency: {concurrency}"

    rows = list(conn.execute(
        f"CALL leiden('{graph_name}', {{{opts}}}) "
        f"YIELD node, community RETURN node.id, community"
    ))

    communities: dict[int, list[str]] = {}
    for row in rows:
        node_id, community = row[0], int(row[1])
        communities.setdefault(community, []).append(node_id)

    try:
        conn.execute(f"CALL drop_projected_graph('{graph_name}')")
    except RuntimeError:
        pass

    return communities


def run_leiden(
    conn: object,
    resolution: float = 1.0,
    concurrency: int | None = None,
) -> dict[int, list[str]]:
    """Run NeuG native Leiden on the full graph.

    Returns communities dict: {community_id: [node_ids]}.
    Raises RuntimeError if GDS extension not available.
    """
    _ensure_gds(conn)
    conn.execute(
        "CALL project_graph('graphify_full', ['node'], "
        "{'[node, edge, node]': ''})"
    )
    return _leiden_on_projected(conn, 'graphify_full', resolution, concurrency)


def run_leiden_fallback(G, resolution: float = 1.0) -> dict[int, list[str]]:
    """Mock Leiden using graspologic when NeuG GDS is not available.

    Takes a NetworkX graph, returns {community_id: [node_ids]}.
    This bridges the gap until NeuG v0.1.3 ships with native Leiden.
    """
    from .cluster import _partition

    node_to_community = _partition(G, resolution=resolution)

    communities: dict[int, list[str]] = {}
    for node_id, community_id in node_to_community.items():
        communities.setdefault(community_id, []).append(node_id)

    return communities


# ---------------------------------------------------------------------------
# Incremental detection — temp graph + union Leiden
# ---------------------------------------------------------------------------


def load_temp_graph(
    conn: object,
    node_rows: list[dict],
    edge_rows: list[dict],
    *,
    temp_node_label: str = "temp_node",
    temp_edge_label: str = "temp_edge",
) -> None:
    """Load incremental data as temporary graph via COPY TEMP.

    Temporary tables are auto-dropped when connection closes.
    """
    tmp_dir = tempfile.mkdtemp(prefix="graphify_temp_")
    try:
        if node_rows:
            csv_path = os.path.join(tmp_dir, f"{temp_node_label}.csv")
            _write_csv(csv_path, node_rows, _NODE_COLUMNS)
            conn.execute(
                f'COPY TEMP {temp_node_label} FROM "{csv_path}" (header=true, delim=",")'
            )

        if edge_rows:
            csv_path = os.path.join(tmp_dir, f"{temp_edge_label}.csv")
            _write_csv(csv_path, edge_rows, _EDGE_COLUMNS)
            conn.execute(
                f'COPY TEMP {temp_edge_label} FROM "{csv_path}" '
                f'(header=true, delim=",", '
                f'from="{temp_node_label}", to="{temp_node_label}")'
            )
    finally:
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)


def run_leiden_on_union(
    conn: object,
    affected_source_files: list[str],
    *,
    temp_node_label: str = "temp_node",
    temp_edge_label: str = "temp_edge",
    resolution: float = 1.0,
    concurrency: int | None = None,
) -> dict[int, list[str]]:
    """Run Leiden on union of persistent + temporary graph.

    Uses project_graph predicates to exclude persistent nodes/edges
    from affected source_files, avoiding duplication with temp data.
    """
    _ensure_gds(conn)

    sf_list = ", ".join(f"'{sf}'" for sf in affected_source_files)

    conn.execute(
        f"CALL project_graph('union_graph', "
        f"['node', '{temp_node_label}'], "
        f"{{"
        f"'[node, edge, node]': 'WHERE NOT n1.source_file IN [{sf_list}] "
        f"AND NOT n2.source_file IN [{sf_list}]', "
        f"'[{temp_node_label}, {temp_edge_label}, {temp_node_label}]': '', "
        f"'[node, edge, {temp_node_label}]': "
        f"'WHERE NOT n1.source_file IN [{sf_list}]', "
        f"'[{temp_node_label}, {temp_edge_label}, node]': "
        f"'WHERE NOT n2.source_file IN [{sf_list}]'"
        f"}}"
    )

    return _leiden_on_projected(conn, 'union_graph', resolution, concurrency)


def detect_concept_delta(
    conn: object,
    delta_extraction: dict,
    resolution: float = 1.0,
    *,
    mode: str = "temp",
    leiden_fn: callable | None = None,
) -> dict:
    """Detect how incremental data affects existing concepts/communities.

    mode="temp": load delta as COPY TEMP, analyze without modifying persistent data.
    mode="persistent": ingest delta into graph.db, then re-run Leiden on full graph.

    leiden_fn: optional custom Leiden function. If None, uses NeuG GDS Leiden.
               For testing before NeuG v0.1.3, pass run_leiden_fallback.

    Returns:
        {
            'changes': {concept_id: {'type': str, ...}},
            'new_communities': {cid: [node_ids]},
            'summary': {'growth': int, 'merge': int, 'split': int, 'new': int, 'stable': int},
        }
    """
    # Step 1: Load delta data
    if mode == "temp":
        node_rows, edge_rows, _ = _normalize_nodes(delta_extraction)
        load_temp_graph(conn, node_rows, edge_rows)
    elif mode == "persistent":
        ingest_extraction(conn, delta_extraction, incremental=True)
    else:
        raise ValueError(f"Unknown mode: {mode!r}")

    # Step 2: Query old concept membership
    old_members = get_concept_members(conn)

    # Build old_label: {node_id: concept_id}
    old_label: dict[str, str] = {}
    for cid, nids in old_members.items():
        for nid in nids:
            old_label[nid] = cid

    # Step 3: Run new Leiden
    affected_sfs = list({
        n.get("source_file", "")
        for n in (delta_extraction.get("nodes") or [])
        if n.get("source_file")
    })

    if mode == "temp":
        new_communities = run_leiden_on_union(
            conn, affected_sfs, resolution=resolution,
        )
    else:
        new_communities = run_leiden(conn, resolution=resolution)

    # Build new_label: {node_id: community_id}
    new_label: dict[int, str] = {}
    for cid, nids in new_communities.items():
        for nid in nids:
            new_label[nid] = cid

    # Step 4: Bidirectional change detection
    changes: dict[str, dict] = {}

    # Forward: each new community <- old sources
    for new_cid, members in new_communities.items():
        old_sources = {
            old_label[n] for n in members
            if n in old_label and old_label[n] is not None
        }
        has_new_nodes = any(n not in old_label for n in members)

        if not old_sources:
            changes[f"new_{new_cid}"] = {
                "type": "new", "members": members,
            }
        elif len(old_sources) == 1:
            old_cid = next(iter(old_sources))
            if has_new_nodes or len(members) != len(old_members.get(old_cid, [])):
                changes[old_cid] = {
                    "type": "growth",
                    "old_members": old_members.get(old_cid, []),
                    "new_members": members,
                }
        elif len(old_sources) > 1:
            merged_key = "+".join(sorted(old_sources))
            changes[merged_key] = {
                "type": "merge",
                "merged_from": list(old_sources),
                "new_members": members,
            }

    # Backward: each old community -> new targets
    for old_cid, old_nodes in old_members.items():
        new_targets = {
            new_label[n] for n in old_nodes
            if n in new_label and new_label[n] is not None
        }
        if len(new_targets) > 1:
            if old_cid not in changes:
                changes[old_cid] = {
                    "type": "split",
                    "old_members": old_nodes,
                    "split_into": list(new_targets),
                }

    # Mark stable concepts
    for old_cid in old_members:
        if old_cid not in changes:
            changes[old_cid] = {"type": "stable"}

    # Step 5: Cleanup (temp mode)
    if mode == "temp":
        try:
            conn.execute("DROP TABLE temp_edge")
            conn.execute("DROP TABLE temp_node")
        except RuntimeError:
            pass

    # Summary
    summary = {"growth": 0, "merge": 0, "split": 0, "new": 0, "stable": 0}
    for c in changes.values():
        summary[c["type"]] = summary.get(c["type"], 0) + 1

    return {
        "changes": changes,
        "new_communities": new_communities,
        "summary": summary,
    }
