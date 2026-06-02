import subprocess
import sys
import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


def run_cmd(cmd, cwd=None):
    try:
        subprocess.run(cmd, shell=True, check=True, cwd=cwd or ROOT_DIR)
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        sys.exit(e.returncode)


def certificados():
    print("Gerando certificados K8s...")
    run_cmd("bash infra/scripts/create_k8s_certs.sh")


def setup_k8s():
    print("Configurando Kubernetes...")
    run_cmd("bash infra/scripts/setup_k8s.sh")


def stop_k8s():
    print("Parando Kubernetes...")
    run_cmd("bash infra/scripts/stop_k8s.sh")


def dev():
    print("Iniciando Steering Service (Dev)...")
    run_cmd("python -m src.steering.server -v")


def client():
    print("Iniciando servidor web do client em http://localhost:8000 ...")
    run_cmd("python -m http.server 8000", cwd=os.path.join(ROOT_DIR, "client"))


def docker_build():
    print("Construindo imagem Docker...")
    run_cmd(
        "docker build -t content-steering/steering-server:latest -f infra/docker/steering.Dockerfile ."
    )
