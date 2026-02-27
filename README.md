# Content Steering with Reinforcement Learning (DASH)

Research project for simulating Content Steering in DASH streaming with multiple decision strategies, controlled non-stationary scenarios, and an automated analysis pipeline.

## Overview

This repository implements:

- A Flask steering service with strategies:
  - `epsilon_greedy`
  - `ucb1`
  - `linucb` (contextual bandit)
  - `oracle_best_choice`
  - `random`
  - `no_steering`
- A dynamic latency oracle with:
  - geographic impact (distance/proximity)
  - temporal jitter
  - latency events (shock/spam)
  - route/network variation and movement smoothing
- 3 simulated cache servers in containers (Caddy) with local HTTPS
- Manual execution (browser + player) and automated batch execution (`run_scenarios.py`)
- Analysis pipeline for log aggregation and graph generation

## Architecture

- `streaming-service/`: Docker cache services (3 nodes)
- `steering-service/`: Flask API + RL strategies + latency oracle
- `client/`: web interface to run browser simulations
- `analysis/`: aggregation and graph scripts
- `run_scenarios.py`: scenario-based experiment orchestrator

## Supported Environments

You can use this project in two ways:

1. Run locally on Linux (your own environment)
2. Run inside the provided VM (dataset already included)

### Option A — Local Linux

#### Prerequisites

- Linux
- Python 3.10+
- Docker + Docker Compose plugin (`docker compose`)
- `mkcert`
- Modern web browser

Optional (recommended):

- Your user in the `docker` group to avoid using `sudo` in all Docker commands.

#### Dataset

Download the dataset from:

- https://drive.google.com/drive/folders/1_Mh1JDoRroikzJnjCsZ-Qgqdbx-XP78N?usp=sharing

Then place the `dataset` folder at the project root, like this:

- `./dataset/Eldorado/4sec/avc/manifest.mpd`

#### Installation

At the project root:

```bash
python -m pip install --upgrade pip
python -m pip install -r steering-service/requirements.txt
```

### Option B — Preconfigured VM (dataset already included)

VM download link:

- https://drive.google.com/file/d/1mCB585muebdJIN6yXXbioIoD1762svy3T/view?usp=sharing

Important note:

- The code inside the VM may be outdated.
- The correct flow is to clone the current repository version and reuse the dataset already present in the VM.

Recommended steps inside the VM:

```bash
cd ~/Documents
git clone https://github.com/alissonpef/Content-Steering content-steering
```

Now move/copy the existing `dataset` folder from the VM into the new clone. Common example:

```bash
cp -r ~/Documents/content-steering-tutorial/dataset ~/Documents/content-steering/
```

Then enter the new project and install dependencies:

```bash
cd ~/Documents/content-steering
python -m pip install --upgrade pip
python -m pip install -r steering-service/requirements.txt
```

Important for both scenarios:

- Containers mount `../dataset` into `/srv` through `streaming-service/docker-compose.yml`.
- If `dataset` is not in the active repository root, the caches start but cannot serve media.

## Local Certificates (HTTPS)

At the project root:

```bash
./create_certs.sh
```

This script generates certificates for:

- `video-streaming-cache-1`
- `video-streaming-cache-2`
- `video-streaming-cache-3`
- `steering-service`

## Start and Stop Caches (Docker)

At the project root:

```bash
./starting_streaming.sh
```

The script:

- starts the 3 cache containers
- detects container IP addresses
- updates `/etc/hosts` with cache hosts and `steering-service`

To stop:

```bash
./stop_streaming.sh
```

## Manual Execution (Interactive Mode)

Use 2 terminals.

### Terminal 1 — Steering service

At the project root:

```bash
python3 steering-service/src/app.py --strategy linucb
```

Other available strategies:

```bash
python3 steering-service/src/app.py --strategy ucb1
python3 steering-service/src/app.py --strategy epsilon_greedy
python3 steering-service/src/app.py --strategy oracle_best_choice
python3 steering-service/src/app.py --strategy random
python3 steering-service/src/app.py --strategy no_steering
```

Useful options:

- `-v` or `--verbose` for detailed logs
- `--log_suffix <suffix>` to distinguish logs

### Terminal 2 — Serve web client

At the project root:

```bash
python3 -m http.server 8001
```

Open in browser:

- `http://127.0.0.1:8001/client/index.html`

## Automated Batch Execution (Scenarios)

The `run_scenarios.py` script runs combinations of strategy × scenario × repetition.

Current scenarios:

1. Baseline (static)
2. Mobility (spatial movement)
3. Latency Shock / Spam (temporal event)
4. Extreme Latency Shock (+1000x)

Example (run everything, 1 repetition per combination):

```bash
python3 run_scenarios.py
```

Additional examples:

```bash
python3 run_scenarios.py --strategies linucb ucb1 epsilon_greedy --runs 3
python3 run_scenarios.py --scenarios 1 2 --runs 2
python3 run_scenarios.py --skip-analysis
python3 run_scenarios.py --skip-docker
```

Log output by scenario:

- `logs/raw/baseline/`
- `logs/raw/mobility/`
- `logs/raw/spam/`
- `logs/raw/spam_extreme/`

## Analysis Pipeline
### 1) Aggregate logs by strategy

Example (baseline):

```bash
python3 analysis/aggregate_logs.py linucb --input_dir logs/raw/baseline --output_dir logs/processed
```

### 2) Individual-run graphs

```bash
python3 analysis/plotting/generate_graphs.py logs/raw/baseline/log_linucb_1.csv
```

### 3) Aggregated graphs

```bash
python3 analysis/plotting/generate_aggregated_graphs.py logs/processed/log_linucb_average.csv
```

### 4) Comparative boxplots

```bash
python3 analysis/plotting/generate_boxplots.py
```

### 5) Time-series comparison across strategies

```bash
python3 analysis/plotting/generate_compare_graphs.py
```

### 6) Server-choice accuracy analysis

```bash
python3 analysis/analyze_server_choices.py
```

Results are saved in `results/` and processed logs in `logs/processed/`.

## Project Structure (Summary)

```text
Content-steering/
├── client/
├── dataset/
├── logs/
│   ├── raw/
│   │   ├── baseline/
│   │   ├── mobility/
│   │   ├── spam/
│   │   └── spam_extreme/
│   └── processed/
├── analysis/
├── steering-service/
├── streaming-service/
└── run_scenarios.py
```

## Quick Troubleshooting

- **Caches start but video does not load:** verify `dataset/` is at the project root.
- **Hostname/certificate error:** run `./create_certs.sh` and `./starting_streaming.sh` again.
- **Docker permission denied:** adjust Docker user permissions or run Docker commands with `sudo`.
- **No module named ...:** reinstall `requirements.txt`.

## Demo

Demo video:

- https://youtu.be/3l2sZNRFYSc
