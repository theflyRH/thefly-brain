"""Whole-brain LIF network built from the FlyWire connectome.

The network is built once and reused for every trial with Brian2's
store()/restore(). Stimulation is delivered by a PoissonGroup wired to the
stimulated neurons, which reproduces Shiu's PoissonInput(weight = w_syn*f_poi)
without rebuilding the Network for each trial.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd
from brian2 import (
    Hz,
    Network,
    NeuronGroup,
    PoissonGroup,
    SpikeMonitor,
    Synapses,
    defaultclock,
    ms,
    mV,
    prefs,
)

from . import neurons as N

DATA = Path(os.environ.get("FLYBRAIN_DATA", Path(__file__).resolve().parents[1] / "data"))
PATH_COMP = DATA / "Completeness_783.csv"
PATH_CON = DATA / "Connectivity_783.parquet"

# Shiu et al. 2024 default parameters (model.py, default_params)
PARAMS = dict(
    v_0=-52 * mV,  # resting potential (Kakaria & de Bivort 2017)
    v_rst=-52 * mV,  # reset potential
    v_th=-45 * mV,  # threshold
    t_mbr=20 * ms,  # membrane time constant
    tau=5 * ms,  # synaptic time constant (Jürgensen et al.)
    t_rfc=2.2 * ms,  # refractory period (Lazar et al.)
    t_dly=1.8 * ms,  # synaptic delay (Paul et al. 2015)
    w_syn=0.275 * mV,  # weight per synapse (free parameter)
    r_poi=150 * Hz,  # Poisson drive of stimulated neurons
    f_poi=250,  # Poisson weight scaling
)
EQS = """
dv/dt = (v_0 - v + g) / t_mbr : volt (unless refractory)
dg/dt = -g / tau               : volt (unless refractory)
rfc                            : second
"""


@dataclass
class TrialResult:
    stimulus: str
    sim_ms: float
    wall_ms: float
    mn9_spikes: int
    active_neurons: int
    spikes_i: np.ndarray
    spikes_t_ms: np.ndarray
    extra: dict = field(default_factory=dict)


class FlyBrain:
    def __init__(self, path_comp: Path = PATH_COMP, path_con: Path = PATH_CON, codegen: str | None = None, shuffle: bool = False):
        """shuffle=True keeps every neuron, weight and parameter but randomly rewires the
        postsynaptic targets: the control that shows the behaviour comes from the wiring."""
        if codegen:
            prefs.codegen.target = codegen
        t0 = time.time()
        self.df_comp = pd.read_csv(path_comp, index_col=0)
        df_con = pd.read_parquet(
            path_con, columns=["Presynaptic_Index", "Postsynaptic_Index", "Excitatory x Connectivity"]
        )
        if shuffle:
            rng = np.random.default_rng(0)
            df_con = df_con.copy()
            df_con["Postsynaptic_Index"] = rng.permutation(df_con["Postsynaptic_Index"].to_numpy())
        self.shuffled = shuffle
        self.n_neurons = len(self.df_comp)
        self.n_synapses = len(df_con)
        self.flyid2i = {int(j): i for i, j in enumerate(self.df_comp.index)}
        self.i2flyid = np.asarray(self.df_comp.index, dtype=np.int64)
        self.dataset = path_comp.stem.replace("Completeness_", "flywire_v")

        # resolve stimulus / readout ids that exist in this dataset
        self.stim_idx: dict[str, np.ndarray] = {}
        for name, ids in N.STIMULI.items():
            present = [self.flyid2i[i] for i in ids if i in self.flyid2i]
            self.stim_idx[name] = np.asarray(present, dtype=np.int64)
        self.mn9_idx = [self.flyid2i[i] for i in (N.MN9_L, N.MN9_R) if i in self.flyid2i]
        self.all_stim_idx = np.unique(np.concatenate([v for v in self.stim_idx.values() if len(v)]))

        # ---- network (Shiu et al. create_model) ----
        neu = NeuronGroup(
            N=self.n_neurons,
            model=EQS,
            method="linear",
            threshold="v > v_th",
            reset="v = v_rst; g = 0 * mV",
            refractory="rfc",
            name="neurons",
            namespace=PARAMS,
        )
        neu.v = PARAMS["v_0"]
        neu.g = 0 * mV
        neu.rfc = PARAMS["t_rfc"]

        syn = Synapses(neu, neu, "w : volt", on_pre="g += w", delay=PARAMS["t_dly"], name="synapses")
        syn.connect(i=df_con["Presynaptic_Index"].values, j=df_con["Postsynaptic_Index"].values)
        syn.w = df_con["Excitatory x Connectivity"].values * PARAMS["w_syn"]
        del df_con

        # Poisson drive: one Poisson source per stimulable neuron, rate set per trial.
        # Equivalent to Shiu's PoissonInput(target_var='v', N=1, weight=w_syn*f_poi).
        self.poi = PoissonGroup(len(self.all_stim_idx), rates=0 * Hz, name="poisson")
        poi_syn = Synapses(self.poi, neu, on_pre="v += w_poi", namespace={"w_poi": PARAMS["w_syn"] * PARAMS["f_poi"]}, name="poisson_syn")
        poi_syn.connect(i=np.arange(len(self.all_stim_idx)), j=self.all_stim_idx)

        self.neu = neu
        self.syn = syn
        self.mon = SpikeMonitor(neu, name="spikes")
        self.net = Network(neu, syn, self.poi, poi_syn, self.mon)
        self.net.store("rest")
        self.build_s = time.time() - t0

    # ------------------------------------------------------------------
    def run_trial(
        self,
        stimulus: str,
        duration_ms: float = 1000,
        rate_hz: float | None = None,
        chunk_ms: float = 10,
        on_chunk: Callable[[float, np.ndarray, bool], None] | None = None,
        eat_threshold_spikes: int = 1,
        stop_after_mn9_spikes: int | None = None,
        min_ms: float = 100,
    ) -> TrialResult:
        """Stimulate `stimulus` neurons with Poisson input and record every spike.

        on_chunk(t_ms, spiking_indices, mn9_fired) is called after each chunk so the
        caller can stream activity while the simulation is still running.
        """
        idx = self.stim_idx.get(stimulus)
        if idx is None:
            raise ValueError(f"unknown stimulus {stimulus}")
        self.net.restore("rest")
        rate = PARAMS["r_poi"] if rate_hz is None else rate_hz * Hz
        rate_arr = np.zeros(len(self.all_stim_idx)) * Hz
        stim_pos = np.searchsorted(self.all_stim_idx, idx)
        rate_arr[stim_pos] = rate
        self.poi.rates = rate_arr
        self.neu.rfc[idx] = 0 * ms  # Shiu: no refractory period for Poisson targets

        t0 = time.time()
        n_seen = 0
        steps = int(round(duration_ms / chunk_ms))
        mn9 = set(self.mn9_idx)
        mn9_so_far = 0
        ran_ms = 0.0
        for k in range(steps):
            self.net.run(chunk_ms * ms)
            ran_ms = (k + 1) * chunk_ms
            i_all = self.mon.i[:]
            new = np.asarray(i_all[n_seen:], dtype=np.int64)
            n_seen = len(i_all)
            fired = bool(len(new)) and bool(mn9.intersection(new.tolist()))
            if fired:
                mn9_so_far += int(np.isin(new, self.mn9_idx).sum())
            if on_chunk is not None:
                on_chunk(ran_ms, np.unique(new), fired)
            # the decision is made once MN9 fires repeatedly: stop early
            if stop_after_mn9_spikes and mn9_so_far >= stop_after_mn9_spikes and ran_ms >= min_ms:
                break
        duration_ms = ran_ms
        wall = (time.time() - t0) * 1000

        spikes_i = np.asarray(self.mon.i[:], dtype=np.int64)
        spikes_t = np.asarray(self.mon.t[:] / ms, dtype=np.float32)
        mn9_spikes = int(np.isin(spikes_i, self.mn9_idx).sum())
        return TrialResult(
            stimulus=stimulus,
            sim_ms=duration_ms,
            wall_ms=wall,
            mn9_spikes=mn9_spikes,
            active_neurons=int(len(np.unique(spikes_i))),
            spikes_i=spikes_i,
            spikes_t_ms=spikes_t,
            extra={"eat": mn9_spikes >= eat_threshold_spikes, "stimulated": int(len(idx))},
        )


def spike_rates(res: TrialResult, i2flyid: np.ndarray, top: int = 20) -> pd.Series:
    """Firing rate (Hz) of the most active neurons, indexed by FlyWire id."""
    counts = pd.Series(res.spikes_i).value_counts()
    rates = counts / (res.sim_ms / 1000)
    rates.index = i2flyid[rates.index]
    return rates.head(top)
