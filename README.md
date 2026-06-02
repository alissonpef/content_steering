# Content Steering with Reinforcement Learning (DASH)

Research project for simulating Content Steering in DASH streaming with multiple decision strategies, using a Kubernetes-native environment to evaluate performance against real-world network latency.

## Overview

This repository implements:

- A FastAPI-based steering service with strategies:
  - `epsilon_greedy`
  - `ucb1`
  - `linucb` (contextual bandit)
  - `thompson_sampling` (contextual Thompson Sampling)
  - `ppo_hybrid` (hybrid PPO policy for bitrate + steering)
  - `sac_hybrid` (hybrid SAC policy for bitrate + steering)
  - `oracle_best_choice`
  - `random`
  - `no_steering`
- A Kubernetes-native architecture (using Kind) that utilizes real cluster latencies to train and evaluate RL algorithms.
- 3 simulated cache servers (Delivery Nodes) using Caddy for local HTTPS.
- Analysis pipeline for log aggregation and graph generation.

## Architecture

        +-------------------+         +-------------------------------------+
        |   Browser (Host)  |         |            Kubernetes (Kind)        |
        | (localhost:5000)  |         |                                     |
        +--------+----------+         |   +-----------------------------+   |
                 | proxy              |   |    dash-client (Nginx)      |   |
                 v                    |   |      (Port 80)              |   |
        +-------------------+         |   +-----+-----------------+-----+   |
        |      Gateway      |-------->|         |                 |         |
        | (Nginx :5000->80) |         |  /steering/         /node[1-3]/     |
        +-------------------+         |         |                 |         |
                                      |         v                 v         |
                                      |  +---------------+ +--------------+ |
                                      |  |steering-server| |delivery-nodes| |
                                      |  | (FastAPI SVC) | | (Caddy SVC)  | |
                                      |  +---------------+ +--------------+ |
                                      +-------------------------------------+

- `steering-service/`: FastAPI + RL strategies + real latency monitoring.
- `client/`: Web interface to run browser simulations via Dash.js.
- `analysis/`: Aggregation and graph generation scripts for experiment logging.
- `delivery-nodes/`, `gateway/`, `manifests/`: K8s configurations and deployment manifests.

## Prerequisites

* **Linux / WSL2**
* **Python 3.12+**
* **Docker**
* **Kind** (Kubernetes IN Docker)
* **kubectl**
* **mkcert** (For local HTTPS generation)

## Dataset

Download the dataset from:

- https://drive.google.com/drive/folders/1_Mh1JDoRroikzJnjCsZ-Qgqdbx-XP78N?usp=sharing

Place the `dataset` folder at the project root, like this:

- `./dataset/Eldorado/4sec/avc/manifest.mpd`

## Quick Start (Kubernetes Mode)

The project runs exclusively in a Kind (Kubernetes) cluster to simulate real network dynamics and caching behaviors.

### 1. Start the environment
Bring up the entire cluster and services with the setup script, specifying your desired strategy:
```bash
./infra/scripts/setup_k8s.sh linucb
```
*(Other strategies: `ucb1`, `epsilon_greedy`, `ppo_hybrid`, `sac_hybrid`, `thompson_sampling`)*

### 2. Accessing the Interface
Access the Gateway from your browser:
http://localhost:5000

If needed, forward the port manually:
```bash
kubectl port-forward deployment/gateway 5000:80
```

### 3. Stop the Environment
Remove the cluster:
```bash
./infra/scripts/stop_k8s.sh
```

## Analysis Pipeline

### 1) Aggregate logs by strategy

```bash
python3 analysis/aggregate_logs.py linucb --input_dir data/logs/raw/baseline --output_dir data/logs/aggregated
```

### 2) Individual-run graphs

```bash
python3 analysis/plotting/generate_graphs.py data/logs/raw/baseline/log_linucb_1.csv
```

### 3) Comparative boxplots

```bash
python3 analysis/plotting/generate_boxplots.py
```

### 4) Server-choice accuracy analysis

```bash
python3 analysis/analyze_server_choices.py
```

Results are saved in `data/results/` and aggregated logs in `data/logs/aggregated/`.

## Demo

Demo video of a previous version:
- https://www.youtube.com/watch?v=HVMiex63daY
