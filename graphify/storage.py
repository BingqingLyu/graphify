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
import json
import os
import re
import tempfile
import warnings
from pathlib import Path

import yaml

from .build import _FILE_TYPE_SYNONYMS, _normalize_id, _norm_source_file
from .validate import VALID_FILE_TYPES

# ---------------------------------------------------------------------------
# DDL — unified schema (4 tables)
# ---------------------------------------------------------------------------

_NODE_DDL = """CREATE NODE TABLE IF NOT EXISTS node (
    id STRING PRIMARY KEY, label STRING, type STRING,
    source_file STRING, source_location STRING)"""

_CONCEPT_DDL = """CREATE NODE TABLE IF NOT EXISTS concept (
    id STRING PRIMARY KEY, name STRING, type STRING,
    description STRING, source STRING, tags STRING)"""

_EDGE_DDL = """CREATE REL TABLE IF NOT EXISTS edge (
    FROM node TO node,
    relation STRING, confidence STRING,
    confidence_score DOUBLE, source_file STRING, weight DOUBLE)"""

_BELONGS_DDL = """CREATE REL TABLE IF NOT EXISTS belongs_to (
    FROM node TO concept)"""

_LINKS_DDL = """CREATE REL TABLE IF NOT EXISTS links_to (
    FROM concept TO concept)"""

# ---------------------------------------------------------------------------
# Column definitions for CSV output
# ---------------------------------------------------------------------------

_NODE_COLUMNS = ["id", "label", "type", "source_file", "source_location"]
_EDGE_COLUMNS = ["from_id", "to_id", "relation", "confidence",
                 "confidence_score", "source_file", "weight"]
_CONCEPT_COLUMNS = ["id", "name", "type", "description", "source", "tags"]
_BELONGS_COLUMNS = ["node_id", "concept_id"]
_LINKS_COLUMNS = ["from_id", "to_id"]


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
    """Execute DDL for the unified schema (5 tables).

    create_tables=True: run CREATE TABLE statements.
    create_tables=False: no-op (kept for API compatibility).
    """
    if create_tables:
        for ddl in (_NODE_DDL, _CONCEPT_DDL, _EDGE_DDL, _BELONGS_DDL, _LINKS_DDL):
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
    """Write Concept nodes + BELONGS_TO edges + LINKS_TO edges to NeuG.

    concepts: list of dicts, each with:
        id: str — concept identifier
        name: str — concept name
        type: str — concept type (optional, default "")
        description: str — concept description (optional, default "")
        source: str — provenance: 'leiden', 'wiki', 'manual', etc. (optional, default "leiden")
        tags: list[str] — tags (optional, default [])
        members: list[str] — node IDs that belong to this concept
        links: list[str] — concept IDs that this concept links to
        citations: list[str] — source file paths cited by this concept
    """
    # Clean up old Leiden concepts and belongs_to edges to avoid duplicates
    # Only clean Leiden concepts, not Wiki or manual concepts
    try:
        conn.execute("MATCH (n:node)-[b:belongs_to]->(c:concept {source: 'leiden'}) DELETE b")
        conn.execute("MATCH (c:concept {source: 'leiden'}) DETACH DELETE c")
    except Exception:
        pass  # Tables might not exist yet

    concept_rows: list[dict] = []
    belongs_rows: list[dict] = []
    links_rows: list[dict] = []

    for c in concepts:
        cid = c.get("id", "")
        if not cid:
            continue
        concept_rows.append({
            "id": cid,
            "name": c.get("name", cid),
            "type": c.get("type", ""),
            "description": c.get("description", ""),
            "source": c.get("source", "leiden"),
            "tags": json.dumps(c.get("tags", []), ensure_ascii=False),
        })
        for nid in c.get("members", []):
            nid_norm = _normalize_id(nid)
            if nid_norm:
                belongs_rows.append({
                    "node_id": nid_norm,
                    "concept_id": cid,
                })
        # citations: resolve source_file paths to node IDs
        for sf in c.get("citations", []):
            try:
                rows = list(conn.execute(
                    "MATCH (n:node) WHERE n.source_file = $sf RETURN n.id",
                    parameters={"sf": sf},
                ))
                for r in rows:
                    belongs_rows.append({
                        "node_id": r[0],
                        "concept_id": cid,
                    })
            except RuntimeError:
                pass
        # links: concept -> concept
        for target_id in c.get("links", []):
            links_rows.append({
                "from_id": cid,
                "to_id": target_id,
            })

    tmp_dir = tempfile.mkdtemp(prefix="graphify_concepts_")
    try:
        # Write concept nodes
        if concept_rows:
            csv_path = os.path.join(tmp_dir, "concept.csv")
            _write_csv(csv_path, concept_rows, _CONCEPT_COLUMNS)
            _copy_csv(conn, csv_path, "concept")
        # Write belongs_to edges
        if belongs_rows:
            csv_path = os.path.join(tmp_dir, "belongs_to.csv")
            _write_csv(csv_path, belongs_rows, _BELONGS_COLUMNS)
            _copy_csv(conn, csv_path, "belongs_to",
                       from_table="node", to_table="concept")
        # Write links_to edges
        if links_rows:
            csv_path = os.path.join(tmp_dir, "links_to.csv")
            _write_csv(csv_path, links_rows, _LINKS_COLUMNS)
            _copy_csv(conn, csv_path, "links_to",
                       from_table="concept", to_table="concept")
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

    Projects both persistent (node) and temporary (temp_node) tables
    into a single projected graph without WHERE-based filtering.
    NeuG's Cypher parser does not support string literals inside
    project_graph predicate values, so we project all nodes and
    let Leiden handle any duplicates naturally.
    """
    _ensure_gds(conn)

    conn.execute(
        f"CALL project_graph('union_graph', "
        f"['node', '{temp_node_label}'], "
        f"{{"
        f"'[node, edge, node]': '', "
        f"'[{temp_node_label}, {temp_edge_label}, {temp_node_label}]': '', "
        f"'[node, edge, {temp_node_label}]': '', "
        f"'[{temp_node_label}, {temp_edge_label}, node]': ''"
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
               For testing, pass a callable that accepts (conn, resolution) and
               returns {community_id: [node_ids]}.

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


# ---------------------------------------------------------------------------
# Wiki impact analysis — incremental effect on wiki knowledge
# ---------------------------------------------------------------------------


def _get_concept_links(conn: object) -> set[tuple[str, str]]:
    """Query all LINKS_TO edges between concepts.

    Returns a set of (from_concept_id, to_concept_id) tuples.
    """
    result: set[tuple[str, str]] = set()
    try:
        rows = conn.execute(
            "MATCH (c1:concept)-[:links_to]->(c2:concept) "
            "RETURN c1.id, c2.id"
        )
        for row in rows:
            result.add((row[0], row[1]))
    except RuntimeError:
        pass
    return result


def _get_wiki_concepts(conn: object) -> dict[str, dict]:
    """Query all concepts with their metadata.

    Returns {concept_id: {name, type, description, source, tags}}.
    """
    result: dict[str, dict] = {}
    try:
        rows = conn.execute(
            "MATCH (c:concept) "
            "RETURN c.id, c.name, c.type, c.description, c.source, c.tags"
        )
        for row in rows:
            cid, name, ctype, desc, source, tags = row
            result[cid] = {
                "name": name or cid,
                "type": ctype or "",
                "description": desc or "",
                "source": source or "leiden",
                "tags": tags or "[]",
            }
    except RuntimeError:
        pass
    return result


def detect_wiki_impact(
    conn: object,
    delta_extraction: dict,
    resolution: float = 1.0,
    *,
    mode: str = "persistent",
    leiden_fn: callable | None = None,
) -> dict:
    """Analyze how incremental raw data affects existing wiki knowledge.

    Two-layer analysis:
    Layer 1 (structural): Load delta data, run Leiden, compare new
    communities with existing concepts to detect split / growth / merge /
    dissolved / stable / new concept candidates. Also detect new / stale
    links between concepts.

    Layer 2 (optional, done by caller): LLM naming for new concept
    candidates and new link rationales.

    mode="persistent" (default): ingest delta into graph.db, then re-run
    Leiden on full graph.
    mode="temp": load delta as COPY TEMP, analyze without modifying
    persistent data. Currently unsupported — requires NeuG GDS Leiden
    to support multi-node-type projected graphs.

    leiden_fn: optional custom Leiden function for testing. If None,
    uses NeuG GDS Leiden. Accepts (conn, resolution) and returns
    {community_id: [node_ids]}.

    Returns:
        {
            'concept_changes': {concept_id: {'type': str, ...}},
            'new_concept_candidates': [{'community_id': int, ...}],
            'link_changes': {'new_links': [...], 'stale_links': [...]},
            'structural_context': {concept_id: {'type': str}},
            'new_communities': {cid: [node_ids]},
            'summary': {'split': int, 'growth': int, 'merge': int, ...},
        }
    """
    # Step 1: Load delta data
    if mode == "persistent":
        ingest_extraction(conn, delta_extraction, incremental=True)
    elif mode == "temp":
        raise NotImplementedError(
            "temp mode requires NeuG GDS Leiden to support multi-node-type "
            "projected graphs. Use mode='persistent' (default)."
        )
    else:
        raise ValueError(f"Unknown mode: {mode!r}")

    # Step 2: Query existing concepts + members + links
    old_concepts = _get_wiki_concepts(conn)
    old_members = get_concept_members(conn)
    old_links = _get_concept_links(conn)

    # Build node → concept_id lookup
    node_to_concept: dict[str, str] = {}
    for cid, nids in old_members.items():
        for nid in nids:
            node_to_concept[nid] = cid

    # Step 3: Run Leiden on full graph (with delta ingested)
    if leiden_fn is not None:
        new_communities = leiden_fn(conn, resolution)
    else:
        new_communities = run_leiden(conn, resolution=resolution)

    # Build node → new_community lookup
    node_to_new_comm: dict[str, int] = {}
    for cid, nids in new_communities.items():
        for nid in nids:
            node_to_new_comm[nid] = cid

    # Step 4: Detect concept changes
    # Backward: each old concept → new targets (split / growth / dissolved / stable)
    concept_changes: dict[str, dict] = {}

    for old_cid, old_nodes in old_members.items():
        new_targets: dict[int, list[str]] = {}
        for nid in old_nodes:
            new_cid = node_to_new_comm.get(nid)
            if new_cid is not None:
                new_targets.setdefault(new_cid, []).append(nid)

        if not new_targets:
            concept_changes[old_cid] = {
                "type": "dissolved",
                "old_members": old_nodes,
            }
        elif len(new_targets) == 1:
            new_cid, matched = next(iter(new_targets.items()))
            new_comm_members = new_communities.get(new_cid, [])
            has_growth = any(n not in old_nodes for n in new_comm_members)
            if has_growth or len(matched) != len(old_nodes):
                concept_changes[old_cid] = {
                    "type": "growth",
                    "old_members": old_nodes,
                    "new_members": new_comm_members,
                }
            else:
                concept_changes[old_cid] = {"type": "stable"}
        else:
            concept_changes[old_cid] = {
                "type": "split",
                "old_members": old_nodes,
                "split_into": {
                    str(cid): matched for cid, matched in new_targets.items()
                },
            }

    # Forward: detect merges (multiple concepts' members now in same community)
    comm_to_concepts: dict[int, set[str]] = {}
    for old_cid, old_nodes in old_members.items():
        for nid in old_nodes:
            new_cid = node_to_new_comm.get(nid)
            if new_cid is not None:
                comm_to_concepts.setdefault(new_cid, set()).add(old_cid)

    for new_cid, concepts in comm_to_concepts.items():
        if len(concepts) > 1:
            for cid in concepts:
                prev = concept_changes.get(cid, {})
                if prev.get("type") in ("stable", "growth"):
                    entry: dict = {
                        "type": "merge",
                        "merged_with": sorted(concepts - {cid}),
                        "new_community": new_cid,
                    }
                    if prev.get("type") == "growth":
                        entry["new_members"] = prev.get("new_members", [])
                    concept_changes[cid] = entry

    # Step 5: Detect new concept candidates
    # A community where >50% of nodes don't belong to any existing concept
    new_concept_candidates: list[dict] = []

    for new_cid, members in new_communities.items():
        known = sum(1 for n in members if n in node_to_concept)
        novelty_ratio = 1.0 - (known / len(members)) if members else 0.0

        if novelty_ratio > 0.5:
            new_concept_candidates.append({
                "community_id": new_cid,
                "members": sorted(members),
                "novelty_ratio": round(novelty_ratio, 3),
                "known_concepts": sorted(set(
                    node_to_concept[n] for n in members if n in node_to_concept
                )),
            })

    # Step 6: Detect link changes between concepts
    # New links: concepts whose members now share the same new community
    # (suggesting structural connection) but don't have a links_to edge
    co_occurring: dict[tuple[str, str], int] = {}
    for new_cid, members in new_communities.items():
        concepts_in_comm: set[str] = set()
        for n in members:
            cid = node_to_concept.get(n)
            if cid is not None:
                concepts_in_comm.add(cid)
        for c1 in sorted(concepts_in_comm):
            for c2 in sorted(concepts_in_comm):
                if c1 < c2:
                    key = (c1, c2)
                    co_occurring[key] = co_occurring.get(key, 0) + 1

    new_links: list[dict] = []
    for (c1, c2), count in co_occurring.items():
        if (c1, c2) not in old_links and (c2, c1) not in old_links:
            new_links.append({
                "from": c1,
                "to": c2,
                "co_occurrence": count,
            })

    # Stale links: links_to edges where concepts no longer co-occur
    stale_links: list[dict] = []
    for (c1, c2) in old_links:
        if (c1, c2) not in co_occurring and (c2, c1) not in co_occurring:
            stale_links.append({
                "from": c1,
                "to": c2,
            })

    # Step 7: Structural context (lightweight summary of structural changes)
    structural_context = {
        cid: {"type": c["type"]}
        for cid, c in concept_changes.items()
    }

    # Summary
    summary = {
        "split": sum(1 for c in concept_changes.values() if c["type"] == "split"),
        "growth": sum(1 for c in concept_changes.values() if c["type"] == "growth"),
        "merge": sum(1 for c in concept_changes.values() if c["type"] == "merge"),
        "dissolved": sum(1 for c in concept_changes.values() if c["type"] == "dissolved"),
        "stable": sum(1 for c in concept_changes.values() if c["type"] == "stable"),
        "new": len(new_concept_candidates),
        "new_links": len(new_links),
        "stale_links": len(stale_links),
    }

    return {
        "concept_changes": concept_changes,
        "new_concept_candidates": new_concept_candidates,
        "link_changes": {
            "new_links": new_links,
            "stale_links": stale_links,
        },
        "structural_context": structural_context,
        "new_communities": new_communities,
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# Wiki import — OKF + graph.json
# ---------------------------------------------------------------------------

_FRONTMATTER_RE = re.compile(r"^---\n([\s\S]*?)\n---\n?([\s\S]*)$")
_LINK_RE = re.compile(r"\]\(([^)\s]+\.md)(?:#[A-Za-z0-9_\-]*)?\)")
_SKIP_FILES = {"index.md", "log.md"}


def _parse_frontmatter(text: str) -> tuple[dict, bool]:
    """Returns (meta_dict, has_valid_type)."""
    m = _FRONTMATTER_RE.match(text)
    if not m:
        return {}, False
    try:
        meta = yaml.safe_load(m.group(1))
        if not isinstance(meta, dict):
            return {}, False
    except yaml.YAMLError:
        return {}, False
    return meta, bool(meta.get("type"))


def _extract_links(body: str, doc_dir: Path, bundle_root: Path) -> list[str]:
    """Extract concept IDs from markdown links."""
    out: list[str] = []
    seen: set[str] = set()
    bundle_root_resolved = bundle_root.resolve()
    for m in _LINK_RE.finditer(body):
        target = m.group(1)
        if "://" in target or target.startswith("/"):
            continue
        try:
            resolved = (doc_dir / target).resolve().relative_to(bundle_root_resolved)
        except ValueError:
            continue
        rel = resolved.as_posix()
        if rel.endswith(".md"):
            rel = rel[:-3]
        if rel and rel not in seen:
            seen.add(rel)
            out.append(rel)
    return out


def _extract_citations(body: str) -> list[str]:
    """Parse the # Citations section (OKF §8) and return source file paths."""
    citations: list[str] = []
    m = re.search(r"^# Citations\s*\n([\s\S]*?)(?:\n# |\Z)", body, re.MULTILINE)
    if not m:
        return citations
    section = m.group(1)
    for line in section.split("\n"):
        line = line.strip().lstrip("- ")
        if not line or "://" in line:
            continue
        link_m = re.match(r"\[.*?\]\(([^)]+)\)", line)
        if link_m:
            path = link_m.group(1).lstrip("/")
        else:
            path = line.strip("`").strip()
        if path:
            citations.append(path)
    return citations


def _humanize_slug(slug: str) -> str:
    """'tables/orders' -> 'Tables / Orders'"""
    return " / ".join(
        p.replace("_", " ").replace("-", " ").title()
        for p in slug.split("/")
    )


def parse_okf_bundle(bundle_path: str | Path) -> list[dict]:
    """Parse OKF bundle (directory of .md files) into concepts list.

    Returns: [{"id": "...", "name": "...", "type": "...", "description": "...",
               "source": "wiki", "tags": [...], "members": [],
               "links": [...], "citations": [...]}]
    """
    bundle = Path(bundle_path)
    concepts: list[dict] = []
    for md_path in sorted(bundle.rglob("*.md")):
        if md_path.name in _SKIP_FILES:
            continue
        rel = md_path.relative_to(bundle)
        concept_id = str(rel.with_suffix(""))
        text = md_path.read_text(encoding="utf-8")
        meta, has_type = _parse_frontmatter(text)
        if not has_type:
            warnings.warn(f"Skipping {rel}: missing 'type' in frontmatter")
            continue
        fm_match = _FRONTMATTER_RE.match(text)
        body = fm_match.group(2) if fm_match else text
        tags = meta.get("tags") or []
        if not isinstance(tags, list):
            tags = [str(tags)]
        concepts.append({
            "id": concept_id,
            "name": str(meta.get("title") or _humanize_slug(concept_id)),
            "type": str(meta.get("type", "")),
            "description": str(meta.get("description", "")),
            "source": "wiki",
            "tags": [str(t) for t in tags],
            "members": [],
            "links": _extract_links(body, md_path.parent, bundle),
            "citations": _extract_citations(body),
        })
    return concepts


def parse_graph_json(path: str | Path) -> list[dict]:
    """Parse graph.json + .graphify_labels.json into concepts list."""
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    links_key = "edges" if "edges" in data else "links"

    labels_path = p.parent / ".graphify_labels.json"
    labels: dict[str, str] = {}
    if labels_path.exists():
        labels = json.loads(labels_path.read_text(encoding="utf-8"))

    communities: dict[int, list[str]] = {}
    community_names: dict[int, str] = {}
    for node in data.get("nodes", []):
        cid = node.get("community")
        if cid is not None:
            communities.setdefault(int(cid), []).append(node["id"])
            cname = node.get("community_name")
            if cname:
                community_names[int(cid)] = cname

    concepts: list[dict] = []
    for cid, members in sorted(communities.items()):
        name = (
            labels.get(str(cid))
            or community_names.get(cid)
            or f"Community {cid}"
        )
        concepts.append({
            "id": f"concept_{cid}",
            "name": name,
            "type": "community",
            "description": "",
            "source": "leiden",
            "tags": [],
            "members": members,
            "links": [],
            "citations": [],
        })
    return concepts


def _detect_format(path: Path) -> str:
    if path.is_file() and path.suffix == ".json":
        return "graph-json"
    elif path.is_dir():
        return "okf"
    else:
        raise ValueError(f"Cannot detect format for {path}")


def import_wiki(
    conn: object,
    path: str | Path,
    format: str = "auto",
) -> int:
    """Import Wiki data into NeuG concept table.

    format="auto": detect by path (file -> graph-json, directory -> okf)
    Returns: number of concepts imported.
    """
    p = Path(path)
    fmt = _detect_format(p) if format == "auto" else format

    if fmt == "graph-json":
        concepts = parse_graph_json(p)
    elif fmt == "okf":
        concepts = parse_okf_bundle(p)
    else:
        raise ValueError(f"Unknown format: {fmt!r}")

    ingest_concepts(conn, concepts)
    return len(concepts)
