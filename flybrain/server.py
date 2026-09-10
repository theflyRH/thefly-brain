"""WebSocket server exposing the connectome simulation to the Node orchestrator.

Protocol (JSON):
  -> {"type":"hello","neurons":N,"synapses":S,"dataset":"flywire_v783","mock":false}
  <- {"type":"stimulate","trialId":"..","stimulus":"sugar","durationMs":1000}
  -> {"type":"frame","trialId":"..","frame":{"tMs":10,"spiking":[...brian indices...],"mn9":false}}
  -> {"type":"result","trialId":"..","mn9Spikes":n,"eat":bool,"activeNeurons":n,"simMs":..,"wallMs":..}
  -> {"type":"error","trialId":"..","msg":".."}

Run:  python -m flybrain.server            (real connectome, builds the network at startup)
      python -m flybrain.server --mock     (no data needed, synthetic activity, for front-end dev)
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import threading
import time

import numpy as np
import websockets

EAT_THRESHOLD = int(os.environ.get("FLYBRAIN_EAT_THRESHOLD_SPIKES", "1"))
# stop the trial once MN9 has fired this many times (0 = always run the full duration)
STOP_AFTER_MN9 = int(os.environ.get("FLYBRAIN_STOP_AFTER_MN9_SPIKES", "4"))
MAX_FRAME_NEURONS = 4000  # cap per-frame payload


class MockBrain:
    """Synthetic activity so the whole pipeline can run without the 100 MB connectome."""

    n_neurons = 138639
    n_synapses = 15091983
    dataset = "mock"
    build_s = 0.0
    mn9_idx = [0]

    def run_trial(self, stimulus, duration_ms=1000, on_chunk=None, chunk_ms=10, **_):  # noqa: ARG002
        rng = np.random.default_rng()
        t0 = time.time()
        mn9 = 0
        active = set()
        for k in range(int(duration_ms / chunk_ms)):
            time.sleep(0.02)
            n = int(rng.integers(50, 400)) if stimulus != "idle" else int(rng.integers(5, 30))
            idx = rng.integers(0, self.n_neurons, size=n)
            active.update(idx.tolist())
            fired = stimulus == "sugar" and k > 15 and rng.random() < 0.35
            mn9 += fired
            if on_chunk:
                on_chunk((k + 1) * chunk_ms, idx, fired)

        class R:  # noqa: D401
            pass

        r = R()
        r.mn9_spikes, r.active_neurons, r.sim_ms, r.wall_ms = mn9, len(active), duration_ms, (time.time() - t0) * 1000
        r.extra = {"eat": mn9 >= EAT_THRESHOLD}
        return r


async def main(host: str, port: int, mock: bool, codegen: str | None):
    if mock:
        brain = MockBrain()
        print("MOCK brain (no connectome loaded)")
    else:
        from .model import FlyBrain

        print("building whole-brain network from FlyWire v783, this takes a while...")
        brain = FlyBrain(codegen=codegen)
        print(
            f"ready: {brain.n_neurons} neurons, {brain.n_synapses} synapses, built in {brain.build_s:.0f}s; "
            f"stimuli: { {k: len(v) for k, v in brain.stim_idx.items()} }, mn9 idx {brain.mn9_idx}"
        )

    lock = threading.Lock()  # one trial at a time, the network is shared

    async def handler(ws):
        loop = asyncio.get_running_loop()
        await ws.send(
            json.dumps(
                {
                    "type": "hello",
                    "neurons": brain.n_neurons,
                    "synapses": brain.n_synapses,
                    "dataset": brain.dataset,
                    "mock": mock,
                }
            )
        )
        async for raw in ws:
            cmd = json.loads(raw)
            if cmd.get("type") != "stimulate":
                continue
            trial_id = cmd["trialId"]
            stimulus = cmd.get("stimulus", "sugar")
            duration = float(cmd.get("durationMs", 1000))
            q: asyncio.Queue = asyncio.Queue()

            def on_chunk(t_ms, idx, mn9_fired):
                idx = np.asarray(idx)
                if len(idx) > MAX_FRAME_NEURONS:
                    idx = np.random.default_rng().choice(idx, MAX_FRAME_NEURONS, replace=False)
                frame = {"tMs": float(t_ms), "spiking": idx.astype(int).tolist(), "mn9": bool(mn9_fired)}
                loop.call_soon_threadsafe(q.put_nowait, {"type": "frame", "trialId": trial_id, "frame": frame})

            def work():
                with lock:
                    try:
                        r = brain.run_trial(stimulus, duration_ms=duration, on_chunk=on_chunk, eat_threshold_spikes=EAT_THRESHOLD, stop_after_mn9_spikes=STOP_AFTER_MN9 or None)
                        msg = {
                            "type": "result",
                            "trialId": trial_id,
                            "mn9Spikes": int(r.mn9_spikes),
                            "eat": bool(r.extra.get("eat")),
                            "activeNeurons": int(r.active_neurons),
                            "simMs": float(r.sim_ms),
                            "wallMs": float(r.wall_ms),
                        }
                    except Exception as e:  # noqa: BLE001
                        msg = {"type": "error", "trialId": trial_id, "msg": str(e)}
                    loop.call_soon_threadsafe(q.put_nowait, msg)

            threading.Thread(target=work, daemon=True).start()
            while True:
                msg = await q.get()
                await ws.send(json.dumps(msg))
                if msg["type"] in ("result", "error"):
                    print(f"trial {trial_id} {stimulus}: {msg}")
                    break

    async with websockets.serve(handler, host, port, max_size=None):
        print(f"flybrain websocket on ws://{host}:{port}")
        await asyncio.Future()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--codegen", default=os.environ.get("FLYBRAIN_CODEGEN"), help="brian2 codegen target: numpy | cython")
    a = ap.parse_args()
    asyncio.run(main(a.host, a.port, a.mock, a.codegen))
