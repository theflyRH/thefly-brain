#!/usr/bin/env bash
# Downloads the FlyWire v783 tables used by the model, from Shiu et al.'s repository (MIT).
# The underlying FlyWire data is CC BY-NC 4.0: cite Dorkenwald et al. 2024 and Schlegel et al. 2024.
set -euo pipefail
cd "$(dirname "$0")"
curl -fsSL -o Completeness_783.csv    https://raw.githubusercontent.com/philshiu/Drosophila_brain_model/main/Completeness_783.csv
curl -fsSL -o Connectivity_783.parquet https://raw.githubusercontent.com/philshiu/Drosophila_brain_model/main/Connectivity_783.parquet
ls -la Completeness_783.csv Connectivity_783.parquet
