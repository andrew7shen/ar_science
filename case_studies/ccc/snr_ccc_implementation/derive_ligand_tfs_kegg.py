"""
Auto-derive LIGAND_TO_TFS from KEGG signal-transduction pathway graphs.

Design rationale (leakage-clean mechanistic prior):
  KEGG signal-transduction pathways are curated from molecular-biology
  literature — receptor-kinase-substrate relations, complex formation,
  canonical activation/inhibition edges. They do NOT incorporate cytokine-
  perturbation transcriptomic response data (the provenance shared by
  NicheNet / MSigDB / Harmonizome / CytoSig). KEGG is therefore a clean
  mechanistic prior for the ligand->TF mapping.

Pipeline:
  1. Fetch the KEGG human gene list (hsa:ID -> gene symbol).
  2. Fetch KGML for each pathway in the "Signal transduction" category +
     cytokine-receptor interaction + a few adjacents (explicitly listed from
     KEGG brite br08901 'KEGG Pathway Maps'). No hand-picked biology.
  3. Parse each KGML: extract <entry type="gene"> nodes (with symbol
     resolution), extract <relation> edges; merge across pathways into a
     single DiGraph keyed on gene symbols.
  4. Pull full CollecTRI TF source list.
  5. For each benchmark ligand, BFS (cutoff=MAX_DEPTH) tracking distance
     and # shortest paths; restrict reachable set to CollecTRI TFs; rank
     by (dist asc, path-count desc, name asc); take top-k.
  6. Write mapping + union + provenance to ligand_tfs_kegg.json.

Output: eval/ccc_#1_eval/snr_ccc_implementation/ligand_tfs_kegg.json
"""

import json
import pathlib
import time
import urllib.request
from collections import deque
from xml.etree import ElementTree as ET

import networkx as nx


THIS_DIR = pathlib.Path(__file__).parent
CACHE_DIR = THIS_DIR / "cache" / "kegg"
OUT_PATH = THIS_DIR / "ligand_tfs_kegg.json"

KEGG_BASE = "https://rest.kegg.jp"

LIGANDS = [
    "BDNF", "BMP2", "BMP4", "BMP6", "CXCL12", "EGF", "FGF2", "GDF11",
    "HGF", "IFNG", "IL10", "IL13", "IL15", "IL1A", "IL1B", "IL2",
    "IL21", "IL22", "IL3", "IL4", "IL6", "LIF", "LTA", "OSM",
    "TGFB1", "TGFB3", "VEGFA", "WNT3A",
]

MAX_DEPTH = 5
TOP_K = 5

# KEGG pathway IDs under 'Environmental Information Processing' > 'Signal transduction'
# (from KEGG brite br08901, human: hsa) + 'Signaling molecules and interaction'
# subset (cytokine-receptor, cell adhesion, ECM-receptor). Fully enumerated so
# no individual pathway is hand-picked for ligand relevance.
SIGNALING_PATHWAYS = [
    # Signal transduction
    "hsa04010",  # MAPK
    "hsa04012",  # ErbB
    "hsa04014",  # Ras
    "hsa04015",  # Rap1
    "hsa04022",  # cGMP-PKG
    "hsa04024",  # cAMP
    "hsa04064",  # NF-kB
    "hsa04066",  # HIF-1
    "hsa04068",  # FoxO
    "hsa04070",  # Phosphatidylinositol
    "hsa04071",  # Sphingolipid
    "hsa04072",  # Phospholipase D
    "hsa04150",  # mTOR
    "hsa04151",  # PI3K-Akt
    "hsa04152",  # AMPK
    "hsa04310",  # Wnt
    "hsa04330",  # Notch
    "hsa04340",  # Hedgehog
    "hsa04350",  # TGF-beta
    "hsa04370",  # VEGF
    "hsa04371",  # Apelin
    "hsa04390",  # Hippo
    "hsa04392",  # Hippo - multi
    "hsa04630",  # JAK-STAT
    "hsa04668",  # TNF
    "hsa04020",  # Calcium
    # Signaling molecules and interaction
    "hsa04060",  # Cytokine-cytokine receptor interaction
    "hsa04061",  # Viral protein interaction with cytokine/receptor
    "hsa04512",  # ECM-receptor interaction
    "hsa04514",  # Cell adhesion molecules
]


def _http_get(url, retries=3, backoff=2.0):
    last_err = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                return r.read().decode("utf-8")
        except Exception as e:
            last_err = e
            time.sleep(backoff * (attempt + 1))
    raise RuntimeError(f"Failed {url} after {retries} attempts: {last_err}")


def fetch_symbol_map():
    """hsa:<ncbi_id> -> primary gene symbol, via KEGG rest list/hsa."""
    cache = CACHE_DIR / "hsa_symbols.txt"
    if cache.exists():
        text = cache.read_text()
    else:
        print("  Fetching https://rest.kegg.jp/list/hsa (~2 MB)...")
        text = _http_get(f"{KEGG_BASE}/list/hsa")
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache.write_text(text)
    mapping = {}
    for line in text.strip().split("\n"):
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) < 4:
            continue
        kegg_id = parts[0].strip()
        symbols_and_desc = parts[3]
        names_part = symbols_and_desc.split(";", 1)[0]
        symbols = [s.strip() for s in names_part.split(",") if s.strip()]
        if symbols:
            mapping[kegg_id] = symbols[0]
    return mapping


def fetch_kgml(pathway_id):
    cache = CACHE_DIR / f"{pathway_id}.kgml"
    if cache.exists():
        return cache.read_text()
    text = _http_get(f"{KEGG_BASE}/get/{pathway_id}/kgml")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache.write_text(text)
    return text


def parse_kgml(text, sym_map):
    """
    Return (nodes, edges) where:
      nodes: dict[entry_id -> set[symbol]]
      edges: list[(entry1_id, entry2_id)]
    Includes type='gene' entries and expands type='group' into member genes
    (by pulling <component> children).
    """
    root = ET.fromstring(text)
    nodes = {}
    component_groups = {}
    for entry in root.findall("entry"):
        etype = entry.get("type")
        eid = entry.get("id")
        if etype == "gene":
            name = entry.get("name", "")
            symbols = set()
            for kegg_id in name.split():
                sym = sym_map.get(kegg_id.strip())
                if sym:
                    symbols.add(sym)
            if symbols:
                nodes[eid] = symbols
        elif etype == "group":
            comps = [c.get("id") for c in entry.findall("component")]
            component_groups[eid] = comps

    # Resolve groups: union of member-entry symbols.
    for gid, comps in component_groups.items():
        syms = set()
        for cid in comps:
            syms.update(nodes.get(cid, set()))
        if syms:
            nodes[gid] = syms

    edges = []
    for rel in root.findall("relation"):
        e1 = rel.get("entry1")
        e2 = rel.get("entry2")
        if e1 in nodes and e2 in nodes:
            edges.append((e1, e2))
    return nodes, edges


def build_kegg_graph(pathway_ids, sym_map, verbose=True):
    G = nx.DiGraph()
    failed = []
    for pid in pathway_ids:
        try:
            text = fetch_kgml(pid)
        except Exception as e:
            failed.append((pid, str(e)))
            if verbose:
                print(f"  {pid}: skipped ({e})")
            continue
        nodes, edges = parse_kgml(text, sym_map)
        if verbose:
            print(f"  {pid}: {len(nodes)} nodes, {len(edges)} relations")
        for e1, e2 in edges:
            for s in nodes[e1]:
                for t in nodes[e2]:
                    if s != t:
                        G.add_edge(s, t)
    if failed and verbose:
        print(f"  {len(failed)} pathways failed to fetch/parse")
    return G, failed


def bfs_with_path_counts(G, source, cutoff):
    if source not in G:
        return {}, {}
    dist = {source: 0}
    count = {source: 1}
    queue = deque([source])
    while queue:
        u = queue.popleft()
        if dist[u] >= cutoff:
            continue
        for v in G.successors(u):
            if v not in dist:
                dist[v] = dist[u] + 1
                count[v] = count[u]
                queue.append(v)
            elif dist[v] == dist[u] + 1:
                count[v] += count[u]
    return dist, count


def load_collectri_tf_pool():
    import decoupler as dc
    return sorted(dc.op.collectri(organism="human")["source"].unique().tolist())


def derive_topk(G, ligands, tf_pool, top_k, max_depth):
    tf_set = set(tf_pool)
    mapping = {}
    diagnostics = {}
    for lig in ligands:
        dist, count = bfs_with_path_counts(G, lig, cutoff=max_depth)
        reachable = [(tf, dist[tf], count[tf]) for tf in tf_set if tf in dist]
        reachable.sort(key=lambda x: (x[1], -x[2], x[0]))
        mapping[lig] = [tf for tf, _, _ in reachable[:top_k]]
        diagnostics[lig] = {
            "n_reachable_tfs": len(reachable),
            "min_dist": reachable[0][1] if reachable else None,
            "in_graph": lig in G,
        }
    return mapping, diagnostics


def main():
    print("Loading KEGG hsa gene-symbol map...")
    sym_map = fetch_symbol_map()
    print(f"  {len(sym_map)} hsa:<id> -> symbol entries")

    print(f"\nBuilding KEGG signaling graph from {len(SIGNALING_PATHWAYS)} pathways...")
    G, failed = build_kegg_graph(SIGNALING_PATHWAYS, sym_map)
    print(f"  Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} directed edges")

    print("\nLoading full CollecTRI TF pool...")
    tf_pool = load_collectri_tf_pool()
    print(f"  {len(tf_pool)} TFs in CollecTRI (organism=human)")

    print(f"\nDeriving top-{TOP_K} mapping (pool = full CollecTRI, BFS cutoff={MAX_DEPTH})...")
    mapping, diag = derive_topk(G, LIGANDS, tf_pool, TOP_K, MAX_DEPTH)

    union_tfs = sorted({tf for tfs in mapping.values() for tf in tfs})
    n_covered = sum(1 for v in mapping.values() if v)

    print(f"\nLigand coverage: {n_covered}/{len(LIGANDS)}")
    missing = [lig for lig in LIGANDS if not mapping[lig]]
    if missing:
        print(f"  MISSING (no TFs reachable in KEGG graph): {missing}")
    print(f"Union of selected TFs: {len(union_tfs)}")
    print(f"\nMapping:")
    for lig in LIGANDS:
        d = diag[lig]
        note = ""
        if not d["in_graph"]:
            note = " [not in KEGG graph]"
        elif not mapping[lig]:
            note = " [no CollecTRI TF reachable]"
        print(f"  {lig:7s}: {mapping[lig]}{note}")

    meta = {
        "source": "KEGG KGML pathway graphs (rest.kegg.jp)",
        "category": "Signal transduction + Signaling molecules and interaction (KEGG brite br08901)",
        "pathways_used": SIGNALING_PATHWAYS,
        "pathways_failed": [pid for pid, _ in failed],
        "tf_pool_source": "full CollecTRI source list (organism=human)",
        "tf_pool_size": len(tf_pool),
        "max_depth": MAX_DEPTH,
        "top_k": TOP_K,
        "ranking": "distance asc, path-count desc, name asc",
        "leakage_note": (
            "KEGG pathway graphs are curated from molecular-biology literature "
            "(kinase-substrate, complex formation, receptor binding). They do "
            "not incorporate cytokine-perturbation transcriptomic response "
            "data. This is a mechanistic prior, independent of CytoSig source "
            "data (the label generator)."
        ),
        "union_tfs": union_tfs,
        "n_union_tfs": len(union_tfs),
        "ligands": LIGANDS,
        "ligands_missing": missing,
        "mapping": mapping,
        "diagnostics": diag,
    }
    with open(OUT_PATH, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nSaved to {OUT_PATH}")


if __name__ == "__main__":
    main()
