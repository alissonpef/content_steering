import os
import subprocess
import sys

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


def run_cmd(cmd, cwd=None):
    try:
        subprocess.run(cmd, shell=True, check=True, cwd=cwd or ROOT_DIR)
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        sys.exit(e.returncode)


def generate_certificates():
    print("Generating K8s certificates...")
    run_cmd("bash infra/scripts/create_k8s_certs.sh")


def setup_k8s():
    print("Setting up Kubernetes...")
    run_cmd("bash infra/scripts/setup_k8s.sh")


def stop_k8s():
    print("Stopping Kubernetes...")
    run_cmd("bash infra/scripts/stop_k8s.sh")


def dev():
    print("Starting Steering Service (Dev)...")
    run_cmd("python -m src.steering.server -v")


def client():
    print("Starting client web server on http://localhost:8000 ...")
    run_cmd("python -m http.server 8000", cwd=os.path.join(ROOT_DIR, "client"))


def docker_build():
    print("Building Docker image...")
    run_cmd(
        "docker build -t content-steering/steering-server:latest -f infra/docker/steering.Dockerfile ."
    )
