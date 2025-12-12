### README.md

# Content Steering with Reinforcement Learning: VM Simulation

**See it in action!** Watch a video demonstrating the project's functionality:
[▶️ Watch the Demo Video](https://youtu.be/3l2sZNRFYSc)

This project demonstrates the application of Content Steering, using the DASH protocol and Reinforcement Learning (including Epsilon-Greedy, D-UCB, and the contextual bandit **LinUCB**), to optimize cache server selection in a simulated video streaming environment. The testing and simulation environment is configured for execution within the provided VirtualBox VM.

## Virtual Machine (VM) Environment

* **User:** `tutorial`
* **Password:** `tutorial`

### Initial Environment and Project Setup

1.  **Download Pre-configured VM (Optional Base):**
    A VirtualBox VM with base software (Docker, Python, mkcert) is available. **Important: The project code in this VM is outdated.**
    [Download VM via Google Drive](https://drive.google.com/file/d/1mCB585muebdJIN6yXXbioIoD1762svy3T/view?usp=sharing)

    If you choose to use this VM, **DO NOT use the project code that comes with it**. Follow the steps below to get the latest version.

2.  **Requirements for Using the VM:**
    * Oracle VirtualBox installed on your system. More information: [https://www.virtualbox.org/](https://www.virtualbox.org/)

3.  **Getting the Latest Project Code in the VM:**
    After importing the VM and logging in (user: `tutorial`, password: `tutorial`):
    * Open a terminal.
    * We recommend cloning the latest version of the project repository directly into the VM. Navigate to a suitable directory (e.g., `~/Documents/`) and clone the repository:
    ```bash
    cd ~/Documents/
    git clone [https://github.com/alissonpef/Content-Steering](https://github.com/alissonpef/Content-Steering) content-steering
    cd content-steering/
    ```
    This directory, `~/Documents/content-steering/`, will be referred to as the "project root directory".

4.  **Preparing the `dataset` Directory:**
    The video dataset ( `.mp4` files, `manifest.mpd`, etc.) is essential for the simulation.
    * **If you used the pre-configured VM:** The outdated VM contains a `dataset` directory at `~/Documents/content-steering-tutorial/dataset/`.
    * **Copy this `dataset` directory to the newly cloned project directory.**
        Assuming the new project was cloned into `~/Documents/content-steering/` and the old VM project is in `~/Documents/content-steering-tutorial/`:
    ```bash
    cp -r ~/Documents/content-steering-tutorial/dataset/ ~/Documents/content-steering/
    ```
    This ensures that the `content-steering/dataset/` directory contains the necessary video files for the cache servers.

## Step-by-Step Execution (with Updated Code)

Follow these instructions in the project root directory. Two terminals will be required.

### Terminal 1: Prepare and Start Backend Services

1.  **Navigate to the Project Root Directory:**
    ```bash
    cd /home/alisson/Alisson/TCC/Content-steering
    ```
    (Or your equivalent path, e.g., `cd ~/Documents/content-steering/`)

2.  **Generate and Place SSL Certificates:**
    SSL certificates are required to run services over HTTPS. The `create_certs.sh` script uses `mkcert` to generate locally trusted certificates.
    
    **First-time setup (if mkcert not installed):**
    ```bash
    sudo apt install mkcert
    mkcert -install
    ```

    **Execute the Certificate Creation Script:**
    From the project root directory, run for each service:
    ```bash
    ./create_certs.sh video-streaming-cache-1
    ./create_certs.sh video-streaming-cache-2
    ./create_certs.sh video-streaming-cache-3
    ./create_certs.sh steering-service
    ```
    
    **Verification (Optional):**
    Certificates should be created in:
    * **Cache Servers:** `streaming-service/certs/` (e.g., `video-streaming-cache-1.pem`, `video-streaming-cache-1-key.pem`)
    * **Steering Service:** `steering-service/certs/` (e.g., `steering-service.pem`, `steering-service-key.pem`)

3.  **Start Cache Servers and Configure Name Resolution (`/etc/hosts`):**
    The `starting_streaming.sh` script handles starting the cache server Docker containers and automatically updating the `/etc/hosts` file. This ensures proper name resolution for the services.
    ```bash
    sudo ./starting_streaming.sh
    ```
    After the script finishes, it will have:
    * Started the `video-streaming-cache-1`, `video-streaming-cache-2`, and `video-streaming-cache-3` containers.
    * Updated `/etc/hosts` to map these container names to their Docker IPs and `steering-service` to `127.0.0.1`.
    *You can verify the containers are running with `docker ps`.*

4.  **Start the Steering Service (Orchestrator):**
    a.  **Install Python Dependencies (Only the first time or if `requirements.txt` is changed):**
        From the project root directory:
    ```bash
    pip3 install -r steering-service/requirements.txt
    ```

    b.  **Run `app.py` specifying the desired steering strategy:**
        **IMPORTANT:** Execute from the **project root directory** (e.g., `/home/alisson/Alisson/TCC/Content-steering/`).
        Choose **one** of the following commands to start the service:

    * **LinUCB (Contextual Bandit):**
    ```bash
    python3 steering-service/src/app.py --strategy linucb
    ```

    * **D-UCB (Dynamic UCB):**
    ```bash
    python3 steering-service/src/app.py --strategy d_ucb
    ```

    * **UCB1:**
    ```bash
    python3 steering-service/src/app.py --strategy ucb1
    ```

    * **Epsilon-Greedy:**
    ```bash
    python3 steering-service/src/app.py --strategy epsilon_greedy
    ```

    * **Random Selection:**
    ```bash
    python3 steering-service/src/app.py --strategy random
    ```

    * **No Steering:**
    ```bash
    python3 steering-service/src/app.py --strategy no_steering
    ```
    
    * **Optimal Strategy (Oracle Best Choice):**
    ```bash
    python3 steering-service/src/app.py --strategy oracle_best_choice
    ```
    
    * **Additional Options:**
        * `--verbose` or `-v`: Enable detailed debug logs from the service
        * `--log_suffix <suffix>`: Append a suffix to log filenames (e.g., `_testScenario1`)

    The Flask server will start on `https://0.0.0.0:30500` (or HTTP if certificates are missing). Keep this terminal open.

### Terminal 2: Serve the Client Interface (HTML Player)

1.  **Navigate to the Project Root Directory:**
    ```bash
    cd /home/alisson/Alisson/TCC/Content-steering
    ```
    (Or your equivalent path, e.g., `cd ~/Documents/content-steering/`)

2.  **Start a Simple HTTP Server for the HTML:**
    ```bash
    python3 -m http.server 8001
    ```
    Keep this terminal open. The client interface will be accessible at `http://127.0.0.1:8001/client/`.

### Running the Simulation in the Browser

1.  **Access the Player Interface:**
    In the VM's browser, go to: `http://127.0.0.1:8001/client/index.html`.
    *Note: If `Content Steering.html` was renamed to `index.html` in the `client/` directory for easier access, use `http://127.0.0.1:8001/client/index.html`.*

2.  **Load the MPD Manifest:**
    The default URL (`https://video-streaming-cache-1/Eldorado/4sec/avc/manifest.mpd`) should work. Click "**Load MPD**".

3.  **Configure and Start the Simulation:**
    * The HTML interface provides default values for total duration (180s), spam events, and movement events. Adjust these parameters as needed for your experiment.
    * Click "**Start Simulation**".

4.  **Data Collection:**
    During the simulation, log files (e.g., `log_d_ucb_1.csv`) will be populated in the `content-steering/logs/raw/` directory.

### Post-Simulation: Data Processing and Graph Generation

After running your simulations, individual log files (e.g., `log_<strategy_name>_<number>.csv` or `log_<strategy_name>_<suffix>_<number>.csv`) will be in `logs/raw/`. The following steps guide you through processing these logs and generating various graphs. All graph-related scripts are located in the `analysis/` directory, and should be run from the **project root directory** unless specified.

**It is recommended to run `aggregate_logs.py` for all your desired strategies first, before running comparison scripts.**

**Step 1: Aggregate Multiple Log Files for Each Strategy**
Use `analysis/aggregate_logs.py` to combine multiple runs of a strategy into a single "average" log file. This script reads from `logs/raw/` and saves aggregated files into `logs/processed/`. By default, it processes all available simulation time data.

* **To aggregate logs for each strategy (run from project root directory):**
    ```bash
    python3 analysis/aggregate_logs.py linucb
    python3 analysis/aggregate_logs.py d_ucb
    python3 analysis/aggregate_logs.py ucb1
    python3 analysis/aggregate_logs.py epsilon_greedy
    python3 analysis/aggregate_logs.py random
    python3 analysis/aggregate_logs.py oracle_best_choice
    python3 analysis/aggregate_logs.py no_steering
    ```
    This creates files like `logs/processed/log_d_ucb_average.csv`.
* **If you used a `--log_suffix` (e.g., `_myTest`) for a set of runs with the steering service:**
    ```bash
    python3 analysis/aggregate_logs.py d_ucb --suffix_pattern _myTest
    ```

**Step 2: Generate Graphs from Aggregated Log Files (Time Series)**
Use `analysis/plotting/generate_aggregated_graphs.py` to visualize the average behavior (as time series plots) from a single *aggregated* `_average.csv` file. **The script automatically detects the maximum simulation time from the data** and adjusts the X-axis accordingly.

* **To process a specific aggregated log file (examples run from project root directory):**
    ```bash
    # Example for LinUCB (auto-detects time range):
    python3 analysis/plotting/generate_aggregated_graphs.py logs/processed/log_linucb_average.csv

    # Example for D-UCB:
    python3 analysis/plotting/generate_aggregated_graphs.py logs/processed/log_d_ucb_average.csv
    
    # Example for UCB1:
    python3 analysis/plotting/generate_aggregated_graphs.py logs/processed/log_ucb1_average.csv
    
    # Optional: Specify maximum time manually (e.g., 200 seconds):
    python3 analysis/plotting/generate_aggregated_graphs.py logs/processed/log_d_ucb_average.csv --max_time 200
    ```
    Graphs are saved in subdirectories within `results/`.
    *Note: Some strategies like LinUCB might not produce data for "RL Values" or "RL Counts", leading to `Plot X skipped` messages. This is expected behavior and not an error.*

**Step 3: Generate Boxplots for Latency Distribution (Individual and Comparison)**
Use `analysis/plotting/generate_boxplots.py` to visualize the distribution of a chosen metric (e.g., `experienced_latency_ms`) for each strategy individually and all strategies side-by-side. This script reads from `logs/processed/`.

* **To generate boxplots using the default metric (`experienced_latency_ms`):**
    ```bash
    python3 analysis/plotting/generate_boxplots.py
    ```
    * You can also specify other metrics:
        ```bash
        python3 analysis/plotting/generate_boxplots.py --metric experienced_latency_ms_CLIENT
        python3 analysis/plotting/generate_boxplots.py --metric dynamic_best_server_latency
        ```
    Individual and comparison boxplots are saved in `results/boxplots/`.

**Step 4: Generate Comparative Analysis Table for Strategy Accuracy**
Use `analysis/analyze_server_choices.py`. It processes `*_average.csv` files found in `logs/processed/`.
* **Run the script (from project root directory):**
    ```bash
    python3 analysis/analyze_server_choices.py
    ```
    This script outputs a CSV table and an image of the table comparing the accuracy of different strategies. The CSV is saved in `logs/processed/` and the image in `results/analysis/`.

**Step 5: Generate a Single Graph Comparing Average Latencies Across All Strategies (Time Series)**
Use `analysis/plotting/generate_compare_graphs.py`. **The script automatically detects the maximum simulation time from all strategy data** and adjusts the X-axis accordingly.
* **To compare the default metric (`experienced_latency_ms`):**
    ```bash
    python3 analysis/plotting/generate_compare_graphs.py
    ```
    * You can also specify other metrics:
        ```bash
        python3 analysis/plotting/generate_compare_graphs.py --metric experienced_latency_ms_CLIENT
        python3 analysis/plotting/generate_compare_graphs.py --metric dynamic_best_server_latency
        ```
    * **Optional: Specify maximum time manually (e.g., 180 seconds):**
        ```bash
        python3 analysis/plotting/generate_compare_graphs.py --max_time 180
        ```
    The comparison graph is saved in `results/`.

**(Optional) Step 6: Generate Detailed Graphs for Individual Simulation Runs (Time Series)**
Use `analysis/plotting/generate_graphs.py` for a single, *non-aggregated* simulation log file.
* **To process a specific individual log file (example for D-UCB, run from project root directory):**
    ```bash
    python3 analysis/plotting/generate_graphs.py logs/raw/log_d_ucb_1.csv
    ```
    If run without arguments from the project root directory, it attempts to process all non-aggregated logs found in `logs/raw/`. Graphs are saved in subdirectories within `results/`.

---

## Additional Useful Commands

* **`docker ps`**: Lists currently running Docker containers.
* **`docker compose -f ./streaming-service/docker-compose.yml down`**: (Run from project root) Stops and removes cache server containers.
* **`docker compose -f ./streaming-service/docker-compose.yml logs -f <service_name>`**: (Run from project root) Tails logs for a specific cache container (e.g., `video-streaming-cache-1`).
* **`docker stop <container_name_or_id>`**: Stops a specific container.
* **`docker start <container_name_or_id>`**: Starts a specific container.

## Troubleshooting

### How to kill a process on a specific port
If you encounter an error saying a port is already in use (e.g., `Address already in use`), you can find and kill the process using that port.

1.  **Find the process ID (PID) using the port (e.g., 30500):**
    ```bash
    sudo lsof -i :30500
    ```
    Or using `netstat`:
    ```bash
    sudo netstat -nlp | grep :30500
    ```

2.  **Kill the process using the PID:**
    ```bash
    sudo kill -9 <PID>
    ```
    Replace `<PID>` with the actual number found in the previous step.

---

## Future Improvements

This project demonstrates a functional Content Steering implementation with Reinforcement Learning. However, several enhancements could make it more aligned with production streaming systems and the DASH-IF Content Steering specification:

### Quality of Experience (QoE) Metrics
- **Throughput-based optimization**: Currently optimizes latency only; should prioritize throughput and bitrate to reduce rebuffering events, as specified in DASH-IF Content Steering guidelines.
- **ABR metrics logging**: Track bitrate selection, buffer levels, quality switches, and rebuffering events to correlate steering decisions with actual video quality.

### Enhanced Network Simulation
- **Dynamic throughput simulation**: Add bandwidth variability to the oracle (congestion, ISP throttling, peak hours) to create realistic streaming scenarios.
- **Composite QoE reward function**: Replace latency-only reward with ITU-T P.1203-based scoring: throughput (40%), bitrate quality (30%), latency (20%), minus rebuffering penalty.

### Content Steering Specification Alignment
- **Adaptive TTL**: Implement dynamic Time-To-Live (5-60s) based on network stability instead of fixed 5-second intervals to reduce steering overhead.
- **Expanded context vector**: Add throughput, buffer level, and current bitrate to the 9D context (→12D) for richer contextual bandit learning.

### Production-Ready Features
- **Session-level metrics**: Aggregate per-session QoE scores (total rebuffering time, average bitrate, startup delay, quality switches) for professional strategy comparison.
- **Realistic CDN topology**: Expand from 3 to 5+ CDNs with varied capacities to simulate multi-CDN environments (CloudFront, Akamai, Fastly).
