### README.md Atualizado

# Content Steering with Reinforcement Learning: VM Simulation

**See it in action!** Watch a video demonstrating the project's functionality:
[▶️ Watch the Demo Video](https://youtu.be/3l2sZNRFYSc)

This project demonstrates the application of Content Steering, using the DASH protocol and Reinforcement Learning (including Epsilon-Greedy, D-UCB, and the contextual bandit **LinUCB**), to optimize cache server selection in a simulated video streaming environment. The testing and simulation environment is configured for execution within the provided VirtualBox VM.

## Virtual Machine (VM) Environment

*   **User:** `tutorial`
*   **Password:** `tutorial`

### Initial Environment and Project Setup

1.  **Download Pre-configured VM (Optional Base):**
    A VirtualBox VM with base software (Docker, Python, mkcert) is available. **Important: The project code in this VM is outdated.**
    [Download VM via Google Drive](https://drive.google.com/file/d/1mCB585muebdJIN6yXbioIoD1762svy3T/view?usp=sharing)

    If you choose to use this VM, **DO NOT use the project code that comes with it**. Follow the steps below to get the latest version.

2.  **Requirements for Using the VM:**
    *   Oracle VirtualBox installed on your system. More information: [https://www.virtualbox.org/](https://www.virtualbox.org/)

3.  **Getting the Latest Project Code in the VM:**
    After importing the VM and logging in (user: `tutorial`, password: `tutorial`):
    *   Open a terminal.
    *   We recommend cloning the latest version of the project repository directly into the VM. Navigate to a suitable directory (e.g., `~/Documents/`) and clone the repository:
    ```bash
    cd ~/Documents/
    git clone https://github.com/alissonpef/Content-Steering content-steering
    cd content-steering/
    ```
    This directory, `~/Documents/content-steering/`, will be referred to as the "project root directory".

4.  **Preparing the `dataset` Directory:**
    The video dataset ( `.mp4` files, `manifest.mpd`, etc.) is essential for the simulation.
    *   **If you used the pre-configured VM:** The outdated VM contains a `dataset` directory at `~/Documents/content-steering-tutorial/dataset/`.
    *   **Copy this `dataset` directory to the newly cloned project directory.**
        Assuming the new project was cloned into `~/Documents/content-steering/` and the old VM project is in `~/Documents/content-steering-tutorial/`:
    ```bash
    cp -r ~/Documents/content-steering-tutorial/dataset/ ~/Documents/content-steering/
    ```
    This ensures that the `content-steering/dataset/` directory contains the necessary video files for the cache servers.

## Step-by-Step Execution in the VM (with Updated Code)

Follow these instructions **inside the VirtualBox VM**, in the updated project root directory (e.g., `~/Documents/content-steering/`). Two terminals will be required.

### Terminal 1: Prepare and Start Backend Services

1.  **Navigate to the Project Root Directory:**
    (E.g., `cd ~/Documents/content-steering/`)

2.  **Generate and Place SSL Certificates:**
    SSL certificates are required to run services over HTTPS. The `create_certs.sh` script (located in the project root `content-steering/`) uses `mkcert` to generate locally trusted certificates.
    *Ensure `mkcert` is installed and you have run `mkcert -install` once to install its local CA into your system's trust stores.*

    a.  **Execute the Certificate Creation Script:**
        The `create_certs.sh` script is designed to be run from the project root directory (`content-steering/`). It expects a service name as an argument. Run it for each cache server and for the steering service:
    ```bash
    ./create_certs.sh video-streaming-cache-1
    ./create_certs.sh video-streaming-cache-2
    ./create_certs.sh video-streaming-cache-3
    ./create_certs.sh steering-service
    ```
    *Note: `mkcert` may request your user password (`tutorial`) if it needs to interact with system trust stores for the first time or if the local CA installation needs permissions.*

    b.  **Verification (Optional):**
        The `create_certs.sh` script should create and move certificates to the following directories within the project:
        *   **Cache Servers:** `content-steering/streaming-service/certs/` (files like `video-streaming-cache-1.pem`, `video-streaming-cache-1-key.pem`, etc.)
        *   **Steering Service:** `content-steering/steering-service/certs/` (files `steering-service.pem`, `steering-service-key.pem`)

3.  **Start Cache Servers and Configure Name Resolution (`/etc/hosts`):**
    The `starting_streaming.sh` script handles starting the cache server Docker containers and automatically updating the `/etc/hosts` file. This ensures proper name resolution for the services.
    ```bash
    sudo ./starting_streaming.sh
    ```
    After the script finishes, it will have:
    *   Started the `video-streaming-cache-1`, `video-streaming-cache-2`, and `video-streaming-cache-3` containers.
    *   Updated `/etc/hosts` to map these container names to their Docker IPs and `steering-service` to `127.0.0.1`.
    *You can verify the containers are running with `docker ps`.*

4.  **Start the Steering Service (Orchestrator):**
    a.  **Install Python Dependencies (Only the first time or if `requirements.txt` is changed):**
        From the project root directory (`content-steering/`). The `tutorial` password may be requested by `sudo`:
    ```bash
    sudo pip3 install -r steering-service/requirements.txt
    ```

    b.  **Run `app.py` specifying the desired steering strategy:**
        Still in the project root directory (`content-steering/`).
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
    * **Add Options:**
        *   `--verbose` or `-v` for detailed debug logs from the service.
        *   `--log_suffix <suffix>` (e.g., `_testScenario1`) to append a suffix to log filenames for better organization.

    The Flask server will start (ideally on `https://0.0.0.0:30500`). Keep this terminal open.

### Terminal 2: Serve the Client Interface (HTML Player)

1.  **Navigate to the Project Root Directory:**
    (E.g., `cd ~/Documents/content-steering/`)

2.  **Start a Simple HTTP Server for the HTML:**
    ```bash
    python3 -m http.server 8000
    ```
    Keep this terminal open.

### Running the Simulation in the Browser

1.  **Access the Player Interface:**
    In the VM's browser, go to: `http://127.0.0.1:8000/Content%20Steering.html`.
    *If the page does not load CSS/JS correctly, ensure you are accessing `Content%20Steering.html` (with the space encoded) or rename the HTML file to not have spaces (e.g., `ContentSteering.html`) and access that.*

2.  **Load the MPD Manifest:**
    The default URL (`https://video-streaming-cache-1/Eldorado/4sec/avc/manifest.mpd`) should work. Click "**Load MPD**".

3.  **Configure and Start the Simulation:**
    *   The HTML interface provides default values for total duration (180s), spam events, and movement events. Adjust these parameters as needed for your experiment.
    *   Click "**Start Simulation**".

4.  **Data Collection:**
    During the simulation, log files (e.g., `log_d_ucb_1.csv`) will be populated in the `content-steering/Graphics/Logs/` directory.


### Post-Simulation: Data Processing and Graph Generation

After running your simulations, individual log files (e.g., `log_<strategy_name>_<number>.csv` or `log_<strategy_name>_<suffix>_<number>.csv`) will be in `Graphics/Logs/`. The following steps guide you through processing these logs and generating various graphs. All graph-related scripts are located in the `Graphics/` directory.

**First, navigate to the `Graphics` directory from your project root:**
```bash
cd Graphics/
```

**Step 1: Aggregate Multiple Log Files for Each Strategy**
Use `aggregate_logs.py` to combine multiple runs of a strategy into a single "average" log file. This script reads from `Graphics/Logs/` and saves aggregated files into `Graphics/Logs/Average/`. Aggregation is limited to a default of 150 seconds of simulation time (configurable in the script).

*   **To aggregate logs for each strategy (run from `Graphics/` directory):**
    ```bash
    python3 aggregate_logs.py linucb
    python3 aggregate_logs.py d_ucb
    python3 aggregate_logs.py ucb1
    python3 aggregate_logs.py epsilon_greedy
    python3 aggregate_logs.py random
    python3 aggregate_logs.py oracle_best_choice
    python3 aggregate_logs.py no_steering
    ```
    This creates files like `Graphics/Logs/Average/log_d_ucb_average.csv`.
*   **If you used a `--log_suffix` (e.g., `_myTest`) for a set of runs with the steering service:**
    ```bash
    python3 aggregate_logs.py d_ucb --suffix_pattern _myTest
    ```

**Step 2: Generate Graphs from Aggregated Log Files (Time Series)**
Use `Generate_aggregated_graphs.py` to visualize the average behavior (as time series plots) from a single *aggregated* `_average.csv` file. X-axes are limited to 150 seconds (configurable in the script).

*   **To process a specific aggregated log file (examples run from `Graphics/` directory):**
    ```bash
    # Example for LinUCB:
    python3 Generate_aggregated_graphs.py Logs/Average/log_linucb_average.csv

    # Example for D-UCB:
    python3 Generate_aggregated_graphs.py Logs/Average/log_d_ucb_average.csv
    
    # Example for UCB1:
    python3 Generate_aggregated_graphs.py Logs/Average/log_ucb1_average.csv 
    ```
    Graphs are saved in subdirectories within `Graphics/Img/`.

**Step 3: Generate Boxplots for Latency Distribution (Individual and Comparison)**
Use `Generate_boxplots.py` to visualize the distribution of a chosen metric (e.g., `experienced_latency_ms`) for each strategy individually and all strategies side-by-side. This script reads from `Graphics/Logs/Average/`.

*   **To generate boxplots using the default metric (`experienced_latency_ms`):**
    ```bash
    python3 Generate_boxplots.py
    ```
    *   You can also specify other metrics:
        ```bash
        python3 Generate_boxplots.py --metric experienced_latency_ms_CLIENT
        python3 Generate_boxplots.py --metric dynamic_best_server_latency
        ```
    Individual and comparison boxplots are saved in `Graphics/Img/boxplots/`.

**Step 4: Generate Comparative Analysis Table for Strategy Accuracy**
Use `analyze_server_choices.py`. It processes `*_average.csv` files found in `Graphics/Logs/Average/`.
*   **Run the script (from `Graphics/` directory):**
    ```bash
    python3 analyze_server_choices.py
    ```
    This script outputs a CSV table and an image of the table comparing the accuracy of different strategies.

**Step 5: Generate a Single Graph Comparing Average Latencies Across All Strategies (Time Series)**
Use `Generate_compare_graphs.py`. The X-axis is limited to 150 seconds (configurable in the script).
*   **To compare the default metric (`experienced_latency_ms`):**
    ```bash
    python3 Generate_compare_graphs.py
    ```
    *   You can also specify other metrics:
        ```bash
        python3 Generate_compare_graphs.py --metric experienced_latency_ms_CLIENT
        python3 Generate_compare_graphs.py --metric dynamic_best_server_latency
        ```
    The comparison graph is saved in `Graphics/Img/`.

**(Optional) Step 6: Generate Detailed Graphs for Individual Simulation Runs (Time Series)**
Use `Generate_graphs.py` for a single, *non-aggregated* simulation log file.
*   **To process a specific individual log file (example for D-UCB, run from `Graphics/` directory):**
    ```bash
    python3 Generate_graphs.py Logs/log_d_ucb_1.csv
    ```
    If run without arguments from the `Graphics/` directory, it attempts to process all non-aggregated logs in `Graphics/Logs/`. Graphs are saved in subdirectories within `Graphics/Img/`.

---

## Additional Useful Commands

*   **`docker ps`**: Lists currently running Docker containers.
*   **`docker compose -f ./streaming-service/docker-compose.yml down`**: (Run from project root) Stops and removes cache server containers.
*   **`docker compose -f ./streaming-service/docker-compose.yml logs -f <service_name>`**: (Run from project root) Tails logs for a specific cache container (e.g., `video-streaming-cache-1`).
*   **`docker stop <container_name_or_id>`**: Stops a specific container.
*   **`docker start <container_name_or_id>`**: Starts a specific container.