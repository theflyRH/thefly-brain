"""Export neuron soma positions + classes for the web brain view.

Writes web/public/atlas/neurons.json. Index i in the atlas == Brian index i
(row order of Completeness_783.csv), so spike frames can be sent as raw indices.
Uses FlyWire codex coordinates.csv.gz (one position per root id, first entry).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import neurons as N
from .model import DATA, PATH_COMP

OUT = Path(__file__).resolve().parents[2] / "web" / "public" / "atlas" / "neurons.json"


def main(out: Path = OUT):
    comp = pd.read_csv(PATH_COMP, index_col=0)
    ids = comp.index.to_numpy(dtype=np.int64)
    coords = pd.read_csv(DATA / "coordinates.csv.gz")
    coords = coords.drop_duplicates("root_id").set_index("root_id")
    cls = pd.read_csv(DATA / "classification.csv.gz", index_col=0)

    pos = np.zeros((len(ids), 3), dtype=np.float32)
    have = coords.index.intersection(ids)
    parsed = coords.loc[have, "position"].str.strip("[]").str.split(expand=True).astype(np.float32)
    idx = pd.Index(ids).get_indexer(have)
    pos[idx] = parsed.to_numpy()
    missing = np.setdiff1d(np.arange(len(ids)), idx)
    # neurons without a soma position get placed at the centroid, slightly jittered
    if len(missing):
        c = pos[idx].mean(axis=0)
        pos[missing] = c + np.random.default_rng(0).normal(0, 2000, size=(len(missing), 3))

    classes = ["other", "sugar", "bitter", "mn9", "motor", "gustatory", "olfactory", "kenyon", "descending", "visual"]
    cid = {c: i for i, c in enumerate(classes)}
    out_cls = np.zeros(len(ids), dtype=np.int8)
    sc = cls.reindex(ids)
    out_cls[(sc["class"] == "gustatory").to_numpy()] = cid["gustatory"]
    out_cls[(sc["class"] == "olfactory").to_numpy()] = cid["olfactory"]
    out_cls[(sc["class"] == "Kenyon_Cell").to_numpy()] = cid["kenyon"]
    out_cls[(sc["super_class"] == "descending").to_numpy()] = cid["descending"]
    out_cls[(sc["super_class"] == "motor").to_numpy()] = cid["motor"]
    out_cls[(sc["super_class"] == "visual_projection").to_numpy()] = cid["visual"]
    id2i = {int(j): i for i, j in enumerate(ids)}
    sugar = [id2i[i] for i in N.SUGAR_GRN_R if i in id2i]
    bitter = [id2i[i] for i in N.BITTER_GRN_R if i in id2i]
    out_cls[sugar] = cid["sugar"]
    out_cls[bitter] = cid["bitter"]
    mn9 = id2i.get(N.MN9_L, -1)
    if mn9 >= 0:
        out_cls[mn9] = cid["mn9"]

    # nm -> ~um/10 to keep numbers short; the front normalizes anyway
    xyz = np.round(pos / 100).astype(np.int32).ravel().tolist()
    # desktop-fly steering/escape circuit (668 FlyWire v783 neurons): atlas index per circuit neuron
    circuit_idx = []
    cpath = DATA / "circuit.json"
    if cpath.exists():
        circ = json.loads(cpath.read_text())
        circuit_idx = [id2i.get(int(n["id"]), -1) for n in circ["neurons"]]

    atlas = {
        "dataset": "flywire_v783",
        "circuitAtlasIndex": circuit_idx,
        "count": int(len(ids)),
        "xyz": xyz,
        "cls": out_cls.tolist(),
        "classes": classes,
        "mn9Index": int(mn9),
        "sugarIndices": sugar,
        "bitterIndices": bitter,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(atlas, separators=(",", ":")))
    print(f"wrote {out} ({out.stat().st_size/1e6:.1f} MB), {len(ids)} neurons, {len(missing)} without soma position")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else OUT)
