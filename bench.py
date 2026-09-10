"""Build the network and run one sugar trial + one idle trial, print timings and MN9 readout.

  .venv/Scripts/python bench.py [--duration 1000] [--codegen cython]
"""
import argparse
import time

from flybrain.model import FlyBrain, spike_rates

ap = argparse.ArgumentParser()
ap.add_argument("--duration", type=float, default=1000)
ap.add_argument("--codegen", default=None)
ap.add_argument("--stimuli", default="sugar,idle,bitter")
ap.add_argument("--shuffle", action="store_true", help="control: randomly rewire the connectome")
a = ap.parse_args()

t0 = time.time()
b = FlyBrain(codegen=a.codegen, shuffle=a.shuffle)
if a.shuffle:
    print("CONTROL: postsynaptic targets randomly permuted (same neurons, weights and parameters)")
print(f"built in {time.time()-t0:.0f}s: {b.n_neurons} neurons, {b.n_synapses} synapses; stim sizes { {k: len(v) for k, v in b.stim_idx.items()} }")

for stim in a.stimuli.split(","):
    r = b.run_trial(stim, duration_ms=a.duration, on_chunk=lambda t, i, m: print(f"  t={t:.0f}ms spiking={len(i)} mn9={m}") if int(t) % 100 == 0 else None)
    print(f"{stim}: sim {r.sim_ms:.0f} ms in {r.wall_ms/1000:.1f} s wall | MN9 spikes={r.mn9_spikes} eat={r.extra['eat']} active={r.active_neurons}")
    print(spike_rates(r, b.i2flyid, top=10))
