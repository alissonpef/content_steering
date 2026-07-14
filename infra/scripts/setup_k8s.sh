#!/bin/bash

set -e

DATASET_DIR="./dataset"
KIND_CONFIG="./infra/k8s/manifests/kind_config.yaml"
CERTS_SCRIPT="./infra/scripts/create_k8s_certs.sh"
CERTS_MANIFEST="./infra/k8s/manifests/k8s_certs.yaml"
K8S_DEPLOY="./infra/k8s/manifests/k8s_deploy.yaml"
KIND_CLUSTER_NAME="kind"

log() {
  echo -e "\033[1;34m[SETUP]\033[0m $1"
}

error() {
  echo -e "\033[1;31m[ERROR]\033[0m $1"
  exit 1
}

log "Checking dataset in $DATASET_DIR..."
if [ ! -d "$DATASET_DIR" ] || [ -z "$(ls -A $DATASET_DIR)" ]; then
  error "Dataset directory is missing or empty. Please populate $DATASET_DIR before running."
fi

log "Creating data directories..."
mkdir -p ./data/logs/raw ./data/logs/aggregated ./data/results

log "Checking Kind cluster..."
if ! kind get clusters | grep -q "^$KIND_CLUSTER_NAME$"; then
  log "Cluster '$KIND_CLUSTER_NAME' not found. Creating..."
  kind create cluster --config "$KIND_CONFIG"
else
  log "Cluster '$KIND_CLUSTER_NAME' already exists."
  kubectl config use-context "kind-$KIND_CLUSTER_NAME"
fi

log "Running certificate generation script..."
if [ ! -f "$CERTS_SCRIPT" ]; then
  error "Certificate script $CERTS_SCRIPT not found."
fi
chmod +x "$CERTS_SCRIPT"
./"$CERTS_SCRIPT"

log "Applying certificate secrets to the cluster..."
if [ ! -f "$CERTS_MANIFEST" ]; then
  error "Certificate manifest $CERTS_MANIFEST was not generated."
fi
kubectl apply -f "$CERTS_MANIFEST"

log "Building and loading Docker images..."
docker build -t content-steering/steering-server:latest -f ./infra/docker/steering.Dockerfile .
docker build -t content-steering/client:latest -f ./infra/docker/client.Dockerfile .
docker build -t content-steering/gateway:latest -f ./infra/docker/gateway.Dockerfile .
docker build -t content-steering/delivery-node:latest -f ./infra/docker/delivery-node.Dockerfile .

kind load docker-image content-steering/steering-server:latest
kind load docker-image content-steering/client:latest
kind load docker-image content-steering/gateway:latest
kind load docker-image content-steering/delivery-node:latest



log "Applying simulator deployments..."
if [ ! -f "$K8S_DEPLOY" ]; then
  error "Deployment manifest $K8S_DEPLOY not found."
fi
kubectl apply -f "$K8S_DEPLOY"

log "Waiting for pods to be ready..."
kubectl wait --for=condition=Ready pods --all --timeout=300s

log "--------------------------------------------------"
log " SETUP COMPLETED SUCCESSFULLY! "
log "--------------------------------------------------"
log "To access the UI, run:"
log "  kubectl port-forward deployment/gateway 5000:80"
log "Then open: http://localhost:5000"
log "--------------------------------------------------"
