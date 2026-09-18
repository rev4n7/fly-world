"""Fly 3D: run the connectome-driven fly in real time and show it in the browser.

Run:  .venv\\Scripts\\python -m flypac.fly3d_server        then open http://127.0.0.1:8765
      (add --demo to start with food, a female and a predator)

The brain simulation runs here in Python; the web page (web/fly3d/) draws the 3D world and the 3D brain
and sends your clicks back (spawn things, remove circuits, injure the brain). Local only, no internet
needed except to load the three.js graphics library from a CDN.
"""
import argparse
import base64
import queue
import json
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pandas as pd

from . import config as C
from .fly3d import BOX_H, FLY3D_DIR, WORLD, Fly3D
from .flyworld import CIRCUITS, FRAME, SEIZURE_FRAC

WEB = C.ROOT / "web" / "fly3d"
MIME = {".html": "text/html", ".js": "text/javascript", ".css": "text/css", ".json": "application/json"}
POINTS_PER_NEURON = 14
SNAPSHOT_EVERY = 0.045   # s between published states (~22 per second)


class Brain3DMap:
    """Real 3D synapse locations per neuron (data/fly3d/anatomy3d.csv), for drawing and for 3D strokes."""

    def __init__(self, brain):
        a = pd.read_csv(FLY3D_DIR / "anatomy3d.csv")
        n = brain.neurons
        # mirrored right-ear cells: left source reflected across the midline (dataset x)
        midline = json.loads((C.DATA_DIR / "meta.json").read_text())["midline_x"] / 1000.0
        mir = a[a.bodyId.isin([-b for b in n.index if b < 0])].copy()
        mir["bodyId"] = -mir.bodyId
        mir["bx"] = np.rint(2 * midline - mir.bx).astype(int)
        a = pd.concat([a, mir])
        a = a[a.bodyId.isin(n.index)]
        a["col"] = brain.idx[a.bodyId].to_numpy()
        self.col, self.xyz, self.count = a.col.to_numpy(), a[["bx", "by", "bz"]].to_numpy(float), a["count"].to_numpy(float)
        self.total = np.bincount(self.col, weights=self.count, minlength=brain.N)
        top = a.sort_values("count", ascending=False).groupby("col").head(POINTS_PER_NEURON)
        self.points = top[["bx", "by", "bz"]].to_numpy(float)
        self.point_col = top.col.to_numpy()
        self.center = self.points.mean(0)

    def cells_in_ball(self, p, r_bins, frac=0.15):
        inside = ((self.xyz - p) ** 2).sum(1) <= r_bins ** 2
        hit = np.bincount(self.col[inside], weights=self.count[inside], minlength=len(self.total))
        return np.flatnonzero((self.total > 0) & (hit >= frac * self.total))


class Sim:
    def __init__(self, seed=0, demo=False):
        self.world = Fly3D(seed=seed)
        self.map = Brain3DMap(self.world.brain)
        self.cmds = queue.Queue()          # clicks from the page, applied by the sim thread between steps
        self.snapshot = b"{}"             # latest state as JSON, published by the sim thread
        self.paused, self.fast = False, False
        self.strokes = []
        self.msg = ""
        if demo:
            for p in [(12, 14), (28, 26), (14, 28), (27, 12)]:
                self.world.spawn("food", p)
            self.world.spawn("female", (24, 20))
            self.world.spawn("predator", (8, 8, 7))

    def run(self):
        """Sim thread: owns the world. The web threads only read `snapshot` and add to `cmds`,
        so serving the page never pauses the brain."""
        owed, last, last_snap = 0.0, time.perf_counter(), 0.0
        while True:
            while not self.cmds.empty():
                try:
                    self.command(self.cmds.get_nowait())
                except Exception as e:          # a bad click must not kill the simulation
                    self.msg = f"command failed: {e}"
            now = time.perf_counter()
            owed = min(owed + now - last, 5 * FRAME)
            last = now
            if self.paused:
                owed = 0.0
            while owed >= FRAME:
                for _ in range(4 if self.fast else 1):
                    self.world.step()
                owed -= FRAME
            if now - last_snap >= SNAPSHOT_EVERY:
                self.snapshot = json.dumps(self.state()).encode()
                last_snap = now
            time.sleep(0.001)

    # ------------------------------------------------------------------ API
    def brain_info(self):
        b, m = self.world.brain, self.map
        roles = b.neurons.role.fillna("").tolist()
        return {"n": b.N, "roles": roles, "types": b.neurons.type.fillna("").tolist(),
                "points": np.round(m.points - m.center, 1).ravel().tolist(), "point_neuron": m.point_col.tolist(),
                "synapses": int(b.edges.weight.sum())}

    def state(self):
        w = self.world
        b = w.brain
        label, lv = w.behaviour()
        rates = np.clip(b.rates * 2.0, 0, 255).astype(np.uint8)          # 0.5 Hz steps, 127 Hz max
        return {
            "t": w.stats.time, "label": label, "levels": lv, "event": w.event if w.stats.time - w.event_time < 1.0 else "",
            "fly": {"pos": w.pos.tolist(), "heading": w.heading.tolist(), "flying": w.flying, "feeding": bool(w.feeding),
                    "singing": bool(w.singing), "escaping": w.stats.time < w.escape_until, "lift": w.lift},
            "entities": [{"id": e.uid, "kind": e.kind, "pos": e.pos.tolist(), "heading": e.heading.tolist(),
                          "amount": e.amount, "resting": w.stats.time < e.rest_until} for e in w.entities],
            "stats": {"jumps": w.stats.jumps, "caught": w.stats.catches, "eaten": round(w.stats.eaten, 2),
                      "song": round(w.stats.song_time, 1)},
            "motor": {k: round(float(v), 1) for k, v in w.motor.items() if not isinstance(v, (bool, tuple))},
            "circuits": [{"key": k, "label": lab, "behav": beh, "n": int(len(w.circuit_idx[k])),
                          "hz": round(float(b.rates[w.circuit_idx[k]].mean()), 1) if len(w.circuit_idx[k]) else 0.0,
                          "removed": k in w.ablated_circuits} for k, lab, beh, _ in CIRCUITS],
            "injury": {"neck": sorted(w.neck_cut), "syn": w.syn_gain, "inh": w.inh_gain, "decline": w.decline,
                       "removed": int((~b.alive).sum()), "seizure": w.seizure_level(), "seizure_on": w.seizure_level() >= SEIZURE_FRAC,
                       "types": sorted(w.ablated_types), "strokes": self.strokes},
            "alive": base64.b64encode(np.packbits(b.alive)).decode(),
            "rates": base64.b64encode(rates.tobytes()).decode(),
            "paused": self.paused, "fast": self.fast, "msg": self.msg, "world": list(WORLD), "box_h": BOX_H,
        }

    def command(self, c):
        w, op = self.world, c.get("op")
        self.msg = ""
        if op == "spawn":
            w.spawn(c["kind"], (c["x"], c["y"], c.get("z", 6.0)))
        elif op == "remove":
            w.remove_near((c["x"], c["y"]))
        elif op == "clear_arena":
            w.entities.clear()
        elif op == "circuit":
            w.toggle_circuit(c["key"])
        elif op == "type":
            ok = w.toggle_type(c["name"])
            self.msg = (("removed " if c["name"] in w.ablated_types else "restored ") + c["name"]) if ok \
                else f"no cell type '{c['name']}' in the simulation"
        elif op == "neck":
            w.set_neck(set(c["sides"]))
        elif op == "lose":
            k = w.lose_random(float(c["frac"]))
            w.flash(f"{k} NEURONS DESTROYED AT RANDOM")
        elif op == "decline":
            w.decline = not w.decline
        elif op == "gains":
            w.set_gains(syn_gain=c.get("syn"), inh_gain=c.get("inh"))
        elif op == "stroke":
            p = np.array(c["p"], float) + self.map.center
            k = w.stroke(self.map.cells_in_ball(p, float(c["r"])))
            self.strokes.append({"p": c["p"], "r": c["r"]})
            w.flash(f"STROKE: {k} NEURONS DESTROYED")
        elif op == "heal":
            w.clear_ablation()
            self.strokes.clear()
        elif op == "pause":
            self.paused = not self.paused
        elif op == "fast":
            self.fast = not self.fast


def make_handler(sim):
    brain_json = json.dumps(sim.brain_info()).encode()

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, body, ctype="application/json", code=200):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = self.path.split("?")[0]
            if path == "/api/state":
                return self._send(sim.snapshot)
            if path == "/api/brain":
                return self._send(brain_json)
            f = WEB / ("index.html" if path == "/" else path.lstrip("/"))
            if f.resolve().is_relative_to(WEB.resolve()) and f.is_file():
                return self._send(f.read_bytes(), MIME.get(f.suffix, "application/octet-stream"))
            self._send(b"not found", "text/plain", 404)

        def do_POST(self):
            if self.path != "/api/cmd":
                return self._send(b"not found", "text/plain", 404)
            c = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            sim.cmds.put(c)
            self._send(b'{"ok": true}')

    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()
    print("loading the brain (a few seconds) ...")
    sim = Sim(seed=args.seed, demo=args.demo)
    threading.Thread(target=sim.run, daemon=True).start()
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(sim))
    url = f"http://127.0.0.1:{args.port}"     # not "localhost": Windows tries IPv6 first and each request stalls ~2 s
    print(f"Fly 3D running: open {url}   (Ctrl+C to stop)")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
