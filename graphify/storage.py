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
    source_file STRING, source_location STRING,
    leiden_comm INT64 DEFAULT -1)"""

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

_NODE_COLUMNS = ["id", "label", "type", "source_file", "source_location", "leiden_comm"]
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
               columns: list[str] | None = None,
               from_table: str | None = None,
               to_table: str | None = None) -> None:
    """COPY FROM for node or relationship tables.

    When from_table/to_table are None, performs a node COPY.
    Otherwise performs a relationship COPY with endpoint references.

    columns: optional explicit column list for a node COPY. Some NeuG versions
    still validate the CSV's physical column count against the full table width,
    so callers must ensure the CSV includes any ALTER-added columns.
    """
    if from_table and to_table:
        conn.execute(
            f'COPY {table} FROM "{csv_path}" '
            f'(from="{from_table}", to="{to_table}", '
            f'header=true, delim=",", escaping=false)'
        )
    else:
        col_clause = f' ({", ".join(columns)})' if columns else ""
        conn.execute(
            f'COPY {table}{col_clause} FROM "{csv_path}" (header=true, delim=",", escaping=false)'
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
            "leiden_comm": int(node.get("leiden_comm", -1)),
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
        _copy_csv(conn, csv_path, node_table, columns=node_columns)

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

    # --- assign valid warm-start ids for newly/re-written nodes ---
    if node_rows:
        try:
            _max_rows = list(conn.execute(
                "MATCH (n:node) WHERE n.leiden_comm >= 0 RETURN max(n.leiden_comm)"
            ))
            _next_comm = int(_max_rows[0][0]) + 1 if _max_rows and _max_rows[0][0] is not None else 0
            for row in node_rows:
                row["leiden_comm"] = _next_comm
                _next_comm += 1
        except RuntimeError:
            pass

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


def read_communities_from_db(db_path: str) -> dict[int, list[str]] | None:
    """Read communities from graph.db and return in cluster() format.

    Returns {int_community_id: [node_ids]} matching the format of
    graphify.cluster.cluster(), or None if graph.db has no concepts.

    Concept IDs in DB are stored as 'concept_{N}' — this function
    strips the prefix to recover the integer key.
    """
    db, conn = init_db(db_path)
    try:
        ensure_schema(conn, create_tables=False)
        raw = get_concept_members(conn)
    finally:
        close_db(db, conn)

    if not raw:
        return None

    communities: dict[int, list[str]] = {}
    for concept_id, node_ids in raw.items():
        # concept_id format: "concept_0", "concept_1", ...
        if concept_id.startswith("concept_"):
            try:
                cid = int(concept_id[len("concept_"):])
            except ValueError:
                continue
        else:
            continue
        communities[cid] = node_ids
    return communities if communities else None


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


def _ensure_leiden_comm_column(conn: object) -> None:
    """Add leiden_comm column to node table if not present. Idempotent."""
    try:
        conn.execute("ALTER TABLE node ADD leiden_comm INT64 DEFAULT -1;")
    except RuntimeError:
        pass  # Column already exists


def _has_leiden_comm_data(conn: object) -> bool:
    """Check if any node has a valid leiden_comm value (not -1)."""
    try:
        rows = list(conn.execute(
            "MATCH (n:node) WHERE n.leiden_comm >= 0 RETURN count(n)"
        ))
        return rows and rows[0][0] > 0
    except RuntimeError:
        return False


def _write_leiden_comm_to_nodes(
    conn: object,
    communities: dict[int, list[str]],
) -> None:
    """Write Leiden community assignments back to node.leiden_comm property.

    This enables incremental warm-start on subsequent Leiden runs via
    initial_community_property.

    Note: NeuG SET clause does not support parameterized values ($param),
    so community ID is interpolated as a literal integer.
    """
    for comm_id, node_ids in communities.items():
        for nid in node_ids:
            try:
                conn.execute(
                    f"MATCH (n:node) WHERE n.id = $nid SET n.leiden_comm = {int(comm_id)}",
                    parameters={"nid": nid},
                )
            except RuntimeError:
                pass


def _leiden_on_projected(
    conn: object,
    graph_name: str,
    resolution: float,
    concurrency: int | None,
    *,
    initial_community_property: str | None = None,
) -> dict[int, list[str]]:
    """Run Leiden on an already-projected graph, parse results, clean up.

    Returns communities dict: {community_id: [node_ids]}.

    initial_community_property: when set, passes the property name to
        multi_label_leiden for incremental warm-start (NeuG GDS feature).
    """
    opts = f"resolution: {resolution}"
    if concurrency is not None:
        opts += f", concurrency: {concurrency}"
    if initial_community_property:
        opts += f", initial_community_property: '{initial_community_property}'"

    rows = list(conn.execute(
        f"CALL multi_label_leiden('{graph_name}', {{{opts}}}) "
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
    *,
    incremental: bool = False,
    ensure_column: bool = False,
    allow_temporary_writes: bool = True,
    write_back: bool = True,
) -> dict[int, list[str]]:
    """Run NeuG native Leiden on the full graph.

    Returns communities dict: {community_id: [node_ids]}.
    Raises RuntimeError if GDS extension not available.

    incremental: when True, uses initial_community_property='leiden_comm'
        for warm-start Leiden if prior community assignments exist in the
        node table.
    ensure_column: when True, migrates legacy DBs by adding node.leiden_comm
        before running. New graphify DBs create the column in _NODE_DDL, so the
        default stays False to avoid noisy ALTER attempts on fresh schemas.
    allow_temporary_writes: deprecated, kept for backward compatibility.
        NeuG multi_label_leiden handles leiden_comm = -1 natively (treats
        such nodes as unassigned). No temporary writes are needed.
    write_back: when True, writes new community assignments back to
        node.leiden_comm.
    """
    _ensure_gds(conn)
    if ensure_column:
        _ensure_leiden_comm_column(conn)

    # Determine if warm-start is possible
    use_warm_start = incremental and _has_leiden_comm_data(conn)

    # NeuG multi_label_leiden handles leiden_comm = -1 natively: nodes with
    # -1 are treated as unassigned and freely assigned by the algorithm.
    # No temporary community ID assignment is needed.

    conn.execute(
        "CALL project_graph('graphify_full', ['node'], "
        "{'[node, edge, node]': ''})"
    )
    communities = _leiden_on_projected(
        conn, 'graphify_full', resolution, concurrency,
        initial_community_property='leiden_comm' if use_warm_start else None,
    )

    # Write communities back to node properties for the next incremental run.
    if write_back and communities:
        _write_leiden_comm_to_nodes(conn, communities)

    return communities

# ---------------------------------------------------------------------------
# God nodes — most-connected real entities (Cypher equivalent of analyze.god_nodes)
# ---------------------------------------------------------------------------

# Labels that are builtin/mock noise — excluded from god-node ranking.
# Must stay in sync with graphify/analyze.py:_BUILTIN_NOISE_LABELS.
_GOD_NODE_NOISE_LABELS = frozenset({
    "str", "int", "float", "bool", "bytes", "bytearray", "complex", "object",
    "True", "False",
    "MagicMock", "Mock", "AsyncMock", "NonCallableMock",
    "NonCallableMagicMock", "PropertyMock", "patch", "sentinel",
    "Path", "Any", "Optional", "List", "Dict", "Set", "Tuple", "Union",
    "Callable", "Type", "ClassVar", "Final", "Literal", "Protocol",
    "Counter", "defaultdict", "OrderedDict", "datetime", "Enum",
    "os", "sys", "re", "json", "io", "abc", "typing",
})

# JSON key labels that are noise when source_file ends with .json.
_GOD_NODE_JSON_NOISE = frozenset({
    "start", "end", "name", "id", "type", "properties",
    "value", "key", "data", "items", "title", "description", "version",
    "dependencies", "devdependencies", "peerdependencies",
    "optionaldependencies", "bundleddependencies", "bundledependencies",
})


def _is_file_node_row(label: str, source_file: str, degree: int) -> bool:
    """Mirror analyze._is_file_node logic for a flat row."""
    if not label:
        return False
    # File-level hub: label matches source filename
    if source_file:
        fname = source_file.rsplit("/", 1)[-1] if "/" in source_file else source_file
        if label == fname:
            return True
    # Method stub: .method_name()
    if label.startswith(".") and label.endswith("()"):
        return True
    # Isolated function stub: function_name() with degree <= 1
    if label.endswith("()") and degree <= 1:
        return True
    return False


def _is_concept_node_row(source_file: str) -> bool:
    """Mirror analyze._is_concept_node logic for a flat row."""
    if not source_file:
        return True
    # No extension in the last path segment → probably a concept label
    last_seg = source_file.rsplit("/", 1)[-1] if "/" in source_file else source_file
    if "." not in last_seg:
        return True
    return False


def _is_json_key_node_row(label: str, source_file: str) -> bool:
    """Mirror analyze._is_json_key_node logic for a flat row."""
    if not source_file or not source_file.lower().endswith(".json"):
        return False
    return (label or "").strip().lower() in _GOD_NODE_JSON_NOISE


def god_nodes_cypher(conn: object, top_n: int = 10) -> list[dict]:
    """Return the top_n most-connected real entities from graph.db.

    Mirrors graphify.analyze.god_nodes logic:
    - Counts degree (in+out) for each node
    - Excludes file-level hubs, concept nodes, JSON key noise, builtin noise
    - Returns [{id, label, degree}] sorted by degree desc
    """
    # Query all nodes with their degree
    rows = conn.execute(
        "MATCH (n:node)-[e]-(m:node) "
        "RETURN n.id, n.label, n.source_file, count(e) AS deg "
        "ORDER BY deg DESC"
    )

    result: list[dict] = []
    for row in rows:
        nid, label, source_file, deg = row[0], row[1] or "", row[2] or "", row[3]
        # Apply same filters as analyze.god_nodes
        if _is_file_node_row(label, source_file, deg):
            continue
        if _is_concept_node_row(source_file):
            continue
        if _is_json_key_node_row(label, source_file):
            continue
        if label in _GOD_NODE_NOISE_LABELS:
            continue
        result.append({"id": nid, "label": label, "degree": deg})
        if len(result) >= top_n:
            break

    return result


def cohesion_cypher(conn: object, communities: dict[int, list[str]]) -> dict[int, float]:
    """Cohesion per community from graph.db (NeuG equivalent of
    cluster.score_all): intra-community undirected edge count / max possible.

    Reads all node-node edges once and tallies in Python, avoiding per-community
    parameterized IN queries. Directed duplicates (a->b, b->a) collapse to one
    undirected edge to match cluster.cohesion_score's undirected subgraph count.
    """
    node_comm: dict[str, int] = {n: cid for cid, nodes in communities.items() for n in nodes}
    intra: dict[int, set] = {cid: set() for cid in communities}
    try:
        for row in conn.execute("MATCH (a:node)-[e:edge]->(b:node) RETURN a.id, b.id"):
            a, b = row[0], row[1]
            if a == b:
                continue
            ca = node_comm.get(a)
            if ca is not None and ca == node_comm.get(b):
                intra[ca].add(frozenset((a, b)))
    except RuntimeError:
        pass
    result: dict[int, float] = {}
    for cid, nodes in communities.items():
        n = len(nodes)
        if n <= 1:
            result[cid] = 1.0
            continue
        possible = n * (n - 1) / 2
        result[cid] = (len(intra.get(cid, ())) / possible) if possible > 0 else 0.0
    return result


def surprising_connections_cypher(
    conn: object,
    communities: dict[int, list[str]] | None = None,
    top_n: int = 5,
) -> list[dict]:
    """NeuG equivalent of analyze._cross_file_surprises: cross-file edges between
    real entities ranked by a composite surprise score, read from graph.db
    (no NetworkX graph). Reuses analyze's pure helpers (_file_category /
    _top_level_dir / _cross_language) and storage row filters; the scoring is a
    dict-backed copy of analyze._surprise_score so the original stays untouched.

    Only the multi-source cross-file path is implemented (the dominant extract
    case). The single-source betweenness fallback is omitted (returns []).
    """
    from .analyze import _file_category, _top_level_dir, _cross_language

    communities = communities or {}
    node_community: dict[str, int] = {n: cid for cid, nodes in communities.items() for n in nodes}

    labels: dict[str, str] = {}
    sources: dict[str, str] = {}
    try:
        for row in conn.execute("MATCH (n:node) RETURN n.id, n.label, n.source_file"):
            labels[row[0]] = row[1] or row[0]
            sources[row[0]] = row[2] or ""
    except RuntimeError:
        return []

    edges: list[tuple] = []
    degrees: dict[str, int] = {}
    try:
        for row in conn.execute(
            "MATCH (a:node)-[e:edge]->(b:node) RETURN a.id, b.id, e.relation, e.confidence"
        ):
            a, b = row[0], row[1]
            edges.append((a, b, row[2] or "", row[3] or "EXTRACTED"))
            degrees[a] = degrees.get(a, 0) + 1
            degrees[b] = degrees.get(b, 0) + 1
    except RuntimeError:
        return []

    def _score(u, v, relation, conf, u_source, v_source):
        # Dict-backed mirror of analyze._surprise_score (original untouched).
        score = 0
        reasons: list[str] = []
        conf_bonus = {"AMBIGUOUS": 3, "INFERRED": 2, "EXTRACTED": 1}.get(conf, 1)
        cat_u, cat_v = _file_category(u_source), _file_category(v_source)
        suppress = (
            conf == "INFERRED" and relation in ("calls", "uses")
            and (_cross_language(u_source, v_source) or {cat_u, cat_v} == {"code", "doc"})
        )
        if suppress:
            conf_bonus = 0
        score += conf_bonus
        if conf in ("AMBIGUOUS", "INFERRED"):
            reasons.append(f"{conf.lower()} connection - not explicitly stated in source")
        if cat_u != cat_v and not suppress:
            score += 2
            reasons.append(f"crosses file types ({cat_u} \u2194 {cat_v})")
        if _top_level_dir(u_source) != _top_level_dir(v_source) and not suppress:
            score += 2
            reasons.append("connects across different repos/directories")
        cu, cv = node_community.get(u), node_community.get(v)
        if cu is not None and cv is not None and cu != cv and not suppress:
            score += 1
            reasons.append("bridges separate communities")
        if relation == "semantically_similar_to":
            score = int(score * 1.5)
            reasons.append("semantically similar concepts with no structural link")
        du, dv = degrees.get(u, 0), degrees.get(v, 0)
        if min(du, dv) <= 2 and max(du, dv) >= 5:
            score += 1
            peripheral = labels.get(u, u) if du <= 2 else labels.get(v, v)
            hub = labels.get(v, v) if du <= 2 else labels.get(u, u)
            reasons.append(f"peripheral node `{peripheral}` unexpectedly reaches hub `{hub}`")
        return score, reasons

    candidates: list[dict] = []
    for a, b, relation, conf in edges:
        if relation in ("imports", "imports_from", "contains", "method"):
            continue
        la, sa, da = labels.get(a, a), sources.get(a, ""), degrees.get(a, 0)
        lb, sb, db_ = labels.get(b, b), sources.get(b, ""), degrees.get(b, 0)
        if _is_concept_node_row(sa) or _is_concept_node_row(sb):
            continue
        if _is_file_node_row(la, sa, da) or _is_file_node_row(lb, sb, db_):
            continue
        if not sa or not sb or sa == sb:
            continue
        score, reasons = _score(a, b, relation, conf, sa, sb)
        candidates.append({
            "_score": score,
            "source": la,
            "target": lb,
            "source_files": [sa, sb],
            "confidence": conf,
            "relation": relation,
            "why": "; ".join(reasons) if reasons else "cross-file semantic connection",
        })
    candidates.sort(key=lambda x: x["_score"], reverse=True)
    for c in candidates:
        c.pop("_score")
    return candidates[:top_n]



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


def _get_concept_names(conn: object) -> dict[str, str]:
    """Query concept display names from the DB."""
    result: dict[str, str] = {}
    try:
        rows = conn.execute("MATCH (c:concept) RETURN c.id, c.name")
        for row in rows:
            cid, name = str(row[0]), str(row[1] or "")
            if cid and name:
                result[cid] = name
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


def analyze_wiki_impact(
    conn: object,
    resolution: float = 1.0,
    *,
    min_concept_size: int = 1,
    baseline_concepts: list[dict] | None = None,
) -> dict:
    """Run Leiden on current graph.db and compare with existing concepts.

    Precondition: graph.db has been updated with new nodes/edges
    (via extract --no-cluster) but concepts are still from the
    previous clustering pass.

    If no concepts exist yet, all communities are reported as new.

    min_concept_size: ignore baseline concepts with fewer members than
    this threshold. Also filters new concept candidates below this size.
    Default 1 (no filtering).
    baseline_concepts: optional list of concept dicts to use as baseline
    instead of querying graph.db. Each dict should have 'id', 'name',
    'members' (list of node IDs), and optionally 'links' (list of
    target concept IDs). Used when comparing against an external wiki.

    Returns::

        {
            'concept_changes': {concept_id: {'type': str, ...}},
            'new_concept_candidates': [{'community_id': int, ...}],
            'link_changes': {'new_links': [...], 'stale_links': [...]},
            'new_communities': {cid: [node_ids]},
            'summary': {'split': int, 'growth': int, ...},
        }
    """
    # Step 1: Get baseline concepts (from parameter or graph.db)
    if baseline_concepts is not None:
        old_members: dict[str, list[str]] = {}
        concept_names: dict[str, str] = {}
        old_links: set[tuple[str, str]] = set()
        for c in baseline_concepts:
            cid = c.get("id", "")
            if cid:
                old_members[cid] = list(c.get("members", []))
                if c.get("name"):
                    concept_names[cid] = str(c.get("name"))
                for target in c.get("links", []):
                    old_links.add((cid, target))
    else:
        old_members = get_concept_members(conn)
        concept_names = _get_concept_names(conn)
        old_links = _get_concept_links(conn)

    # Build full node set BEFORE min_concept_size filtering (for change detection)
    _all_concept_nodes: set[str] = set()
    for _nids in old_members.values():
        _all_concept_nodes.update(_nids)

    # Build node → concept_id lookup (after filtering by min_concept_size)
    if min_concept_size > 1:
        old_members = {cid: nids for cid, nids in old_members.items() if len(nids) >= min_concept_size}
    node_to_concept: dict[str, str] = {}
    for cid, nids in old_members.items():
        for nid in nids:
            node_to_concept[nid] = cid

    # Step 2: Proceed directly to Leiden with warm-start.
    # Previously an early-return check skipped Leiden when all nodes were
    # covered by existing concepts. This is now unnecessary because:
    # (a) incremental warm-start Leiden produces stable results on unchanged
    #     portions (no non-determinism noise to avoid), and
    # (b) edge-only changes (new call relations, removed imports) affect
    #     community structure without changing node count — the old check
    #     missed these entirely.

    # Step 3: Run Leiden on full graph (raw result, no Python postprocessing)
    # Use incremental warm-start so unchanged portions remain stable —
    # detected changes reflect genuine structural shifts, not Leiden noise.
    new_communities = run_leiden(
        conn,
        resolution=resolution,
        concurrency=1,
        incremental=True,
        ensure_column=False,
        allow_temporary_writes=False,
        write_back=False,
    )

    # Re-index by size for stable comparison (same logic as cluster.py NeuG path)
    _sorted = sorted(
        new_communities.values(),
        key=lambda nodes: (-len(nodes), tuple(sorted(map(str, nodes)))),
    )
    new_communities = {i: sorted(nodes) for i, nodes in enumerate(_sorted)}

    # Build node → new_community lookup
    node_to_new_comm: dict[str, int] = {}
    for cid, nids in new_communities.items():
        for nid in nids:
            node_to_new_comm[nid] = cid

    # Step 3: Detect concept changes (split / growth / dissolved / stable)
    # Stability threshold: if 50%+ of a concept's members stay in the same
    # new community, treat drift as Leiden noise, not a real split.
    # Growth threshold: only report growth if the community grew by >20%.
    #
    # CRITICAL: growth counts only GENUINELY-NEW nodes (not in _all_concept_nodes
    # i.e. not part of any baseline concept), NOT old nodes that drifted in from
    # other concepts. Empirically ~99% of size-based "growth" is old-node drift
    # caused by Leiden's global re-optimization (verified via warm-start test):
    # a 2-file diff produced 124 raw growths but only 2 involved genuinely-new
    # nodes. Filtering to delta nodes removes this noise so reported growth
    # reflects real new code absorbed into an existing concept.
    _STABILITY_THRESHOLD = 0.5
    _GROWTH_MIN_RATIO = 0.2  # at least 20% new members to count as growth
    concept_changes: dict[str, dict] = {}
    dominant_assignments: dict[str, int] = {}

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
            continue

        # Find the dominant new community (where most members ended up)
        dominant_cid = max(new_targets, key=lambda c: len(new_targets[c]))
        dominant_count = len(new_targets[dominant_cid])
        dominant_ratio = dominant_count / len(old_nodes) if old_nodes else 0

        if dominant_ratio >= _STABILITY_THRESHOLD:
            dominant_assignments[old_cid] = dominant_cid
            # Most members stayed together — this is stable or growth, not split.
            # Count only genuinely-new nodes (absent from every baseline concept)
            # to exclude old-node drift noise.
            new_comm_members = new_communities.get(dominant_cid, [])
            new_in_comm = [n for n in new_comm_members if n not in _all_concept_nodes]
            old_drift_in_comm = [n for n in new_comm_members if n in _all_concept_nodes and n not in old_nodes]
            growth_ratio = len(new_in_comm) / len(old_nodes) if old_nodes else 0
            if new_in_comm and growth_ratio >= _GROWTH_MIN_RATIO:
                concept_changes[old_cid] = {
                    "type": "growth",
                    "old_members": old_nodes,
                    "new_members": new_comm_members,
                    "target_community": dominant_cid,
                    "delta_members": sorted(new_in_comm),
                    "old_drift_members": sorted(old_drift_in_comm),
                }
            else:
                concept_changes[old_cid] = {"type": "stable"}
        else:
            # Members genuinely scattered — check for real split
            valid_subs = {
                cid: matched for cid, matched in new_targets.items()
                if len(matched) >= min_concept_size
            }

            if len(valid_subs) >= 2:
                concept_changes[old_cid] = {
                    "type": "split",
                    "old_members": old_nodes,
                    "split_into": {
                        str(cid): matched for cid, matched in valid_subs.items()
                    },
                }
            elif len(valid_subs) == 1:
                dom_cid, dom_matched = next(iter(valid_subs.items()))
                new_comm_members = new_communities.get(dom_cid, [])
                # Growth only when genuinely-new nodes joined (not old-node drift)
                delta_members = [n for n in new_comm_members if n not in _all_concept_nodes]
                old_drift_members = [n for n in new_comm_members if n in _all_concept_nodes and n not in old_nodes]
                if delta_members:
                    concept_changes[old_cid] = {
                        "type": "growth",
                        "old_members": old_nodes,
                        "new_members": new_comm_members,
                        "target_community": dom_cid,
                        "delta_members": sorted(delta_members),
                        "old_drift_members": sorted(old_drift_members),
                    }
                else:
                    concept_changes[old_cid] = {"type": "stable"}
            else:
                concept_changes[old_cid] = {
                    "type": "dissolved",
                    "old_members": old_nodes,
                }

    # Step 3.5: Detect merges / collapses (collapse-first).
    # Multiple concepts whose dominant community is the same new community
    # have effectively merged. When too many concepts pile into one community
    # it is a Leiden collapse (god community), not a genuine merge.
    _MERGE_MAX_CONCEPTS = 10
    _GOD_THRESHOLD = 1000
    merge_groups: dict[int, list[str]] = {}
    for old_cid, dom_cid in dominant_assignments.items():
        merge_groups.setdefault(dom_cid, []).append(old_cid)

    # First pass: identify collapse target communities.
    collapse_targets: set[int] = set()
    for dom_cid, group in merge_groups.items():
        if len(group) < 2:
            continue
        target_size = len(new_communities.get(dom_cid, []))
        if len(group) > _MERGE_MAX_CONCEPTS or target_size >= _GOD_THRESHOLD:
            collapse_targets.add(dom_cid)

    # Second pass: reclassify concepts in each group.
    for dom_cid, group in merge_groups.items():
        if len(group) < 2:
            continue
        is_collapse = dom_cid in collapse_targets
        change_type = "collapse" if is_collapse else "merge"
        target_size = len(new_communities.get(dom_cid, []))
        for old_cid in group:
            existing = concept_changes.get(old_cid, {})
            if existing.get("type") in ("growth", "stable"):
                concept_changes[old_cid] = {
                    "type": change_type,
                    "old_members": old_members[old_cid],
                    "target_community": dom_cid,
                    "target_size": target_size,
                    "group_size": len(group),
                    "merged_with": sorted(c for c in group if c != old_cid),
                }

    # Step 3.6: Catch growth concepts (path B) whose target is a collapse
    # community. These escaped Step 3.5 because they weren't in
    # dominant_assignments (dominant_ratio < 0.5). Also catches growth into
    # any god-sized community (>=1000) even if no other concept targeted it.
    for old_cid, change in concept_changes.items():
        if change.get("type") != "growth":
            continue
        target_comm = change.get("target_community", -1)
        new_members = change.get("new_members") or []
        if target_comm in collapse_targets or len(new_members) >= _GOD_THRESHOLD:
            concept_changes[old_cid] = {
                "type": "collapse",
                "old_members": change.get("old_members", []),
                "target_community": target_comm,
                "target_size": len(new_members),
                "group_size": 0,
                "merged_with": [],
            }

    # Step 4: Detect new concept candidates
    # A community qualifies as a new-concept candidate only when BOTH hold:
    #   (a) novelty_ratio > 0.5 — most members don't belong to any concept in
    #       the (min_concept_size-filtered) baseline lookup, AND
    #   (b) delta_ratio > 0.5 — most members are genuinely-new delta nodes
    #       (absent from EVERY baseline concept, i.e. not in _all_concept_nodes).
    # The delta gate mirrors the growth/link filters: without it, a community
    # made purely of old code that the wiki simply never covered (delta == 0,
    # or whose members only fell out of the lookup due to min_concept_size)
    # would be mislabeled "new". Those are baseline blind spots, not concepts
    # introduced by this change.
    _NEW_CONCEPT_MIN_DELTA_RATIO = 0.5
    new_concept_candidates: list[dict] = []

    for new_cid, members in new_communities.items():
        if not members:
            continue
        known = sum(1 for n in members if n in node_to_concept)
        novelty_ratio = 1.0 - (known / len(members))
        delta_count = sum(1 for n in members if n not in _all_concept_nodes)
        delta_ratio = delta_count / len(members)

        if (
            novelty_ratio > 0.5
            and delta_ratio > _NEW_CONCEPT_MIN_DELTA_RATIO
            and len(members) >= min_concept_size
        ):
            new_concept_candidates.append({
                "community_id": new_cid,
                "members": sorted(members),
                "novelty_ratio": round(novelty_ratio, 3),
                "delta_ratio": round(delta_ratio, 3),
                "known_concepts": sorted(set(
                    node_to_concept[n] for n in members if n in node_to_concept
                )),
            })

    # Step 5: Detect link changes between concepts
    # New links: wiki concepts whose members now share the same new community
    # (suggesting structural connection) but don't have a links_to edge.
    # Filter: only count co-occurrences in communities that contain at least
    # one delta node — pure old-node drift does not constitute a real link.
    co_occurring: dict[tuple[str, str], int] = {}
    for new_cid, members in new_communities.items():
        # Skip communities with no delta nodes (pure drift)
        if not any(n not in _all_concept_nodes for n in members):
            continue
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
    weak_new_links: list[dict] = []
    for (c1, c2), count in co_occurring.items():
        if (c1, c2) not in old_links and (c2, c1) not in old_links:
            link = {
                "from": c1,
                "to": c2,
                "co_occurrence": count,
            }
            if count >= 2:
                new_links.append(link)
            else:
                weak_new_links.append(link)

    # Stale links: links_to edges where concepts no longer co-occur
    stale_links: list[dict] = []
    for (c1, c2) in old_links:
        if (c1, c2) not in co_occurring and (c2, c1) not in co_occurring:
            stale_links.append({
                "from": c1,
                "to": c2,
            })

    # Summary
    summary = {
        "split": sum(1 for c in concept_changes.values() if c["type"] == "split"),
        "growth": sum(1 for c in concept_changes.values() if c["type"] == "growth"),
        "merge": sum(1 for c in concept_changes.values() if c["type"] == "merge"),
        "collapse": sum(1 for c in concept_changes.values() if c["type"] == "collapse"),
        "dissolved": sum(1 for c in concept_changes.values() if c["type"] == "dissolved"),
        "stable": sum(1 for c in concept_changes.values() if c["type"] == "stable"),
        "new": len(new_concept_candidates),
        "new_links": len(new_links),
        "weak_new_links": len(weak_new_links),
        "stale_links": len(stale_links),
    }

    # Community size distribution: baseline (extract-time leiden_comm in DB)
    # vs re-clustered (this run's warm-start Leiden result). Lets the caller
    # see whether re-clustering collapsed into god communities.
    try:
        _bl_rows = list(conn.execute(
            "MATCH (n:node) WHERE n.leiden_comm IS NOT NULL "
            "RETURN n.leiden_comm, count(n) AS sz ORDER BY sz DESC"
        ))
        _bl_sizes = [r[1] for r in _bl_rows]
    except Exception:
        _bl_sizes = []
    _new_sizes = sorted((len(v) for v in new_communities.values()), reverse=True)
    community_distribution = {
        "baseline": {
            "count": len(_bl_sizes),
            "max": _bl_sizes[0] if _bl_sizes else 0,
            "god_count": sum(1 for s in _bl_sizes if s >= _GOD_THRESHOLD),
            "top10": _bl_sizes[:10],
        },
        "reclustered": {
            "count": len(_new_sizes),
            "max": _new_sizes[0] if _new_sizes else 0,
            "god_count": sum(1 for s in _new_sizes if s >= _GOD_THRESHOLD),
            "top10": _new_sizes[:10],
        },
    }

    return {
        "concept_changes": concept_changes,
        "new_concept_candidates": new_concept_candidates,
        "concept_names": concept_names,
        "link_changes": {
            "new_links": new_links,
            "weak_new_links": weak_new_links,
            "stale_links": stale_links,
        },
        "new_communities": new_communities,
        "community_distribution": community_distribution,
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# High-level orchestration helpers (called by __main__.py)
# ---------------------------------------------------------------------------


def neug_sync(
    db_path: str,
    extraction: dict,
    *,
    incremental: bool = False,
    prune_sources: list[str] | None = None,
    root: object | None = None,
    communities: dict[int, list[str]] | None = None,
) -> tuple | None:
    """Sync extraction data (and optionally communities) to graph.db.

    Encapsulates all NeuG write operations for the extract pipeline:
    1. Open/create graph.db
    2. Ingest extraction (nodes + edges)
    3. Optionally write communities as concepts
    4. Close connection

    Returns (db, conn) if caller needs the connection kept open (conn != None
    means NeuG is available), or None if NeuG is not installed.

    If communities is provided, also calls ingest_concepts and closes
    the connection. If communities is None, returns (db, conn) open
    for the caller to pass conn to cluster().
    """
    try:
        db, conn = init_db(db_path)
    except Exception:
        return None

    ensure_schema(conn, create_tables=not incremental)
    ingest_extraction(conn, extraction, incremental=incremental,
                      prune_sources=prune_sources, root=root)

    if communities is not None:
        if communities:  # Non-empty: write concepts
            try:
                ingest_concepts(conn, [
                    {"id": f"concept_{cid}", "name": f"Community {cid}",
                     "source": "leiden", "members": members}
                    for cid, members in communities.items()
                ])
            except Exception:
                pass
        # communities provided (even empty) → close and return None
        close_db(db, conn)
        return None

    # Return open connection for caller (e.g. to pass to cluster)
    return (db, conn)


def run_wiki_impact(
    db_path: str,
    *,
    min_concept_size: int = 1,
    resolution: float = 1.0,
    baseline_path: str | None = None,
    graph_json_path: str | None = None,
    backend: str | None = None,
    model: str | None = None,
    output_format: str = "text",
) -> str:
    """Run wiki-impact analysis end-to-end and return formatted output.

    db_path: path to graph.db
    min_concept_size: filter out concepts with fewer members (default 1)
    resolution: Leiden resolution parameter (default 1.0, lower = larger communities)
    baseline_path: optional external wiki path (auto-detect format)
    graph_json_path: path to graph.json (for LLM naming context)
    backend: LLM backend for naming new concepts (optional)
    model: override LLM model
    output_format: 'text' or 'json'

    Returns formatted string (text report or JSON).
    """
    from pathlib import Path
    import json as _json

    # Load external baseline if specified
    baseline_concepts: list[dict] | None = None
    if baseline_path is not None:
        p = Path(baseline_path)
        if not p.exists():
            raise FileNotFoundError(f"baseline path not found: {baseline_path}")
        fmt = _detect_format(p)
        if fmt == "graph-json":
            baseline_concepts = parse_graph_json(p)
        elif fmt == "okf":
            baseline_concepts = parse_okf_bundle(p)
        else:
            raise ValueError(f"cannot detect wiki format for {baseline_path}")

    # Run analysis directly against the requested DB. wiki-impact uses
    # read-only Leiden flags below, so this path must not alter schema or
    # write node.leiden_comm.
    db, conn = init_db(db_path)
    ensure_schema(conn, create_tables=False)
    try:
        result = analyze_wiki_impact(conn, resolution=resolution, min_concept_size=min_concept_size, baseline_concepts=baseline_concepts)
        # Get god nodes for LLM naming prioritization (same conn)
        gods = god_nodes_cypher(conn)

        # Optional LLM naming — uses conn to read node labels from graph.db
        # (the NeuG backend is the source of truth for the full node set).
        if backend:
            try:
                from graphify.llm import label_communities, detect_backend as _detect_backend
                actual_backend = backend or _detect_backend() or "gemini"

                # Collect all communities that need naming:
                # 1. Changed old concepts (by their current/old members)
                # 2. Split sub-communities
                # 3. New concept candidates
                # Use integer keys for label_communities compatibility
                communities_to_name: dict[int, list[str]] = {}
                key_to_origin: dict[int, str] = {}  # int key -> original id
                next_key = 0

                for cid, info in result["concept_changes"].items():
                    if info["type"] in ("growth", "dissolved", "merge"):
                        communities_to_name[next_key] = info.get("old_members", [])
                        key_to_origin[next_key] = f"change:{cid}"
                        next_key += 1
                    elif info["type"] == "split":
                        communities_to_name[next_key] = info.get("old_members", [])
                        key_to_origin[next_key] = f"change:{cid}"
                        next_key += 1
                        # Also name split sub-communities
                        for sub_cid, sub_members in info.get("split_into", {}).items():
                            communities_to_name[next_key] = sub_members
                            key_to_origin[next_key] = f"split:{cid}:{sub_cid}"
                            next_key += 1

                for c in result.get("new_concept_candidates", []):
                    communities_to_name[next_key] = c["members"]
                    key_to_origin[next_key] = f"new:{c['community_id']}"
                    next_key += 1

                link_endpoint_cids: set[str] = set()
                for link in result.get("link_changes", {}).get("new_links", []):
                    link_endpoint_cids.update([link["from"], link["to"]])
                for link in result.get("link_changes", {}).get("weak_new_links", []):
                    link_endpoint_cids.update([link["from"], link["to"]])
                for cid in sorted(link_endpoint_cids):
                    current_name = result.get("concept_names", {}).get(cid, "")
                    if current_name and not current_name.startswith("Community "):
                        continue
                    members = result.get("concept_changes", {}).get(cid, {}).get("old_members") or []
                    if not members and baseline_concepts is None:
                        members = get_concept_members(conn).get(cid, [])
                    if members:
                        communities_to_name[next_key] = members
                        key_to_origin[next_key] = f"link:{cid}"
                        next_key += 1

                if communities_to_name:
                    labels = label_communities(
                        None, communities_to_name,
                        backend=actual_backend, model=model,
                        gods=gods, conn=conn,
                    )
                    # Build reverse lookup: origin -> name
                    origin_to_name: dict[str, str] = {}
                    for k, origin in key_to_origin.items():
                        if k in labels:
                            origin_to_name[origin] = labels[k]

                    # Apply names back
                    for cid, info in result["concept_changes"].items():
                        key = f"change:{cid}"
                        if key in origin_to_name:
                            info["name"] = origin_to_name[key]
                        if info["type"] == "split":
                            split_names = {}
                            for sub_cid in info.get("split_into", {}):
                                skey = f"split:{cid}:{sub_cid}"
                                if skey in origin_to_name:
                                    split_names[sub_cid] = origin_to_name[skey]
                            if split_names:
                                info["split_names"] = split_names
                    for c in result.get("new_concept_candidates", []):
                        nkey = f"new:{c['community_id']}"
                        c["name"] = origin_to_name.get(nkey, f"Community {c['community_id']}")
                    for cid in link_endpoint_cids:
                        lkey = f"link:{cid}"
                        if lkey in origin_to_name:
                            result.setdefault("concept_names", {})[cid] = origin_to_name[lkey]
            except Exception as _naming_exc:
                import sys as _sys
                print(f"[wiki-impact] LLM naming failed: {_naming_exc}", file=_sys.stderr)
    finally:
        close_db(db, conn)

    # Format output
    if output_format == "json":
        serializable = {
            "concept_changes": result["concept_changes"],
            "new_concept_candidates": result["new_concept_candidates"],
            "concept_names": result.get("concept_names", {}),
            "link_changes": result["link_changes"],
            "structural_context": result.get("structural_context", {}),
            "community_distribution": result.get("community_distribution", {}),
            "summary": result["summary"],
        }
        return _json.dumps(serializable, indent=2, default=str)

    return _format_wiki_impact_text(result)


def _format_wiki_impact_text(result: dict) -> str:
    """Format wiki-impact result as human-readable text."""
    lines: list[str] = []
    _sum = result["summary"]
    _concept_names = result.get("concept_names", {})
    _cd = result.get("community_distribution") or {}
    lines.append("Wiki impact:")
    if _cd:
        _bl = _cd.get("baseline", {})
        _rc = _cd.get("reclustered", {})
        lines.append("  community distribution:")
        lines.append(f"    baseline (extract):  {_bl.get('count', 0)} communities, max={_bl.get('max', 0)}, god(>=1000)={_bl.get('god_count', 0)}  top10={_bl.get('top10', [])}")
        lines.append(f"    re-clustered:        {_rc.get('count', 0)} communities, max={_rc.get('max', 0)}, god(>=1000)={_rc.get('god_count', 0)}  top10={_rc.get('top10', [])}")
    lines.append("  concept changes:")
    lines.append(f"    stable:    {_sum.get('stable', 0)}")
    lines.append(f"    growth:    {_sum.get('growth', 0)}")
    lines.append(f"    merge:     {_sum.get('merge', 0)}")
    lines.append(f"    collapse:  {_sum.get('collapse', 0)}")
    lines.append(f"    split:     {_sum.get('split', 0)}")
    lines.append(f"    dissolved: {_sum.get('dissolved', 0)}")
    lines.append(f"  new concepts:  {_sum.get('new', 0)}")
    lines.append(f"  new links:     {_sum.get('new_links', 0)}")
    if _sum.get('weak_new_links', 0):
        lines.append(f"  weak links:    {_sum.get('weak_new_links', 0)} (co-occurrence=1, hidden)")
    lines.append(f"  stale links:   {_sum.get('stale_links', 0)}")

    # Split details
    splits = [(cid, info) for cid, info in result["concept_changes"].items() if info["type"] == "split"]
    if splits:
        show = splits[:10]
        lines.append(f"  --- split (top {len(show)} of {len(splits)}) ---")
        for cid, info in show:
            name = info.get('name', _concept_names.get(cid, cid))
            old_n = len(info.get('old_members', []))
            into = info.get('split_into', {})
            split_names = info.get('split_names', {})
            lines.append(f"    {name} ({old_n} members) -> {len(into)} sub-communities")
            for sub_cid, sub_members in into.items():
                sub_name = split_names.get(sub_cid, f"sub-{sub_cid}")
                lines.append(f"      -> {sub_name} ({len(sub_members)} members)")

    # Growth details (top 10)
    growths = [(cid, info) for cid, info in result["concept_changes"].items() if info["type"] == "growth"]
    if growths:
        growths.sort(key=lambda x: len(x[1].get('delta_members', [])), reverse=True)
        show = growths[:10]
        lines.append(f"  --- growth (top {len(show)} of {len(growths)}) ---")
        for cid, info in show:
            name = info.get('name', _concept_names.get(cid, cid))
            old_n = len(info.get('old_members', []))
            new_n = len(info.get('new_members', []))
            delta_n = len(info.get('delta_members', []))
            drift_n = len(info.get('old_drift_members', []))
            detail = f"+{delta_n} delta"
            if drift_n:
                detail += f", +{drift_n} old-drift"
            lines.append(f"    {name}: {old_n} -> {new_n} members ({detail})")

    # Dissolved details
    dissolved = [(cid, info) for cid, info in result["concept_changes"].items() if info["type"] == "dissolved"]
    if dissolved:
        show = dissolved[:10]
        lines.append(f"  --- dissolved (top {len(show)} of {len(dissolved)}) ---")
        for cid, info in show:
            name = info.get('name', _concept_names.get(cid, cid))
            lines.append(f"    {name} ({len(info.get('old_members', []))} members lost)")

    # Merge details — group by target community, top 10 groups
    merges = [(cid, info) for cid, info in result["concept_changes"].items() if info["type"] == "merge"]
    if merges:
        lines.append(f"  --- merge ({len(merges)}) ---")
        _by_target: dict[int, list[str]] = {}
        _tgt_sizes: dict[int, int] = {}
        for cid, info in merges:
            tgt = info.get("target_community", -1)
            _by_target.setdefault(tgt, []).append(cid)
            _tgt_sizes[tgt] = info.get("target_size", 0)
        for tgt, group in sorted(_by_target.items(), key=lambda kv: -len(kv[1]))[:10]:
            names = [_concept_names.get(c, c) for c in group]
            lines.append(f"    {', '.join(names)} -> Community {tgt} ({_tgt_sizes.get(tgt, 0)} members)")

    # Collapse details — concepts absorbed into a god community
    collapses = [(cid, info) for cid, info in result["concept_changes"].items() if info["type"] == "collapse"]
    if collapses:
        lines.append(f"  --- collapse ({len(collapses)}) ---")
        _by_target_c: dict[int, list[str]] = {}
        _tgt_sizes_c: dict[int, int] = {}
        for cid, info in collapses:
            tgt = info.get("target_community", -1)
            _by_target_c.setdefault(tgt, []).append(cid)
            _tgt_sizes_c[tgt] = info.get("target_size", 0)
        for tgt, group in sorted(_by_target_c.items(), key=lambda kv: -len(kv[1]))[:10]:
            lines.append(f"    Community {tgt} ({_tgt_sizes_c.get(tgt, 0)} members) absorbed {len(group)} concepts")
        omitted = len(_by_target_c) - min(10, len(_by_target_c))
        if omitted > 0:
            lines.append(f"    ({omitted} more collapse targets)")

    # New concept candidates
    candidates = result.get("new_concept_candidates", [])
    if candidates:
        # Sort by size descending, show top 10
        candidates_sorted = sorted(candidates, key=lambda c: -len(c['members']))
        show = candidates_sorted[:10]
        if len(candidates) > 10:
            lines.append(f"  --- new concept candidates (top 10 of {len(candidates)}) ---")
        else:
            lines.append(f"  --- new concept candidates ({len(candidates)}) ---")
        for c in show:
            name = c.get("name", f"Community {c['community_id']}")
            members = c['members']
            n = len(members)
            # Show first 8 node IDs, truncate the rest
            preview_nodes = members[:8]
            preview = ', '.join(preview_nodes)
            suffix = f" (+{n - 8} more)" if n > 8 else ""
            lines.append(f"    {name} ({n} members): {preview}{suffix}")
        omitted = len(candidates) - len(show)
        if omitted > 0:
            lines.append(f"    ({omitted} more)")
    # Link changes
    lc = result.get("link_changes", {})
    # Build concept name lookup for link display
    _cid_names: dict[str, str] = dict(result.get("concept_names", {}))
    for cid, info in result["concept_changes"].items():
        _cid_names[cid] = info.get('name', _cid_names.get(cid, cid))
    new_links = lc.get("new_links", [])
    if new_links:
        lines.append(f"  --- new links ({len(new_links)}, showing top 10) ---")
        nl_sorted = sorted(new_links, key=lambda x: -x['co_occurrence'])
        for link in nl_sorted[:10]:
            from_name = _cid_names.get(link['from'], link['from'])
            to_name = _cid_names.get(link['to'], link['to'])
            lines.append(f"    {from_name} -> {to_name} (co-occurrence: {link['co_occurrence']})")
    weak_new_links = lc.get("weak_new_links", [])
    if weak_new_links:
        lines.append(f"  --- weak new links omitted ({len(weak_new_links)}, co-occurrence=1) ---")
    stale_links = lc.get("stale_links", [])
    if stale_links:
        lines.append(f"  --- stale links ({len(stale_links)}) ---")
        for link in stale_links[:10]:
            from_name = _cid_names.get(link['from'], link['from'])
            to_name = _cid_names.get(link['to'], link['to'])
            lines.append(f"    {from_name} -> {to_name}")

    return "\n".join(lines)


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
