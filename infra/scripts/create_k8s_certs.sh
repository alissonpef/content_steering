#!/bin/bash

set -e

SERVICES=(
  "delivery-node-1"
  "delivery-node-2"
  "delivery-node-3"
  "steering-server"
)

echo "=================================================="
echo " Verifying the local mkcert CA installation... "
echo "=================================================="
mkcert -install || true
echo ""

for SERVICE_NAME in "${SERVICES[@]}"; do
  DEST_DIR=""

  if [[ "$SERVICE_NAME" == "steering-server" ]]; then
    DEST_DIR="./certs/steering-service"
  elif [[ "$SERVICE_NAME" == "delivery-node-"* ]]; then
    DEST_DIR="./certs/delivery-nodes"
  else
    echo "WARNING: Unknown service name: '$SERVICE_NAME'. Skipping."
    continue
  fi

  echo "--------------------------------------------------"
  echo "  Creating certificate for: $SERVICE_NAME"
  echo "  Destination directory:   $DEST_DIR"
  echo "--------------------------------------------------"

  mkdir -p "$DEST_DIR"

  echo "--> Generating certificate with mkcert..."
  mkcert -cert-file "./${SERVICE_NAME}.pem" -key-file "./${SERVICE_NAME}-key.pem" "$SERVICE_NAME" "$SERVICE_NAME.default.svc.cluster.local" "localhost" 127.0.0.1

  echo "--> Moving certificate files..."
  mv "./${SERVICE_NAME}.pem" "$DEST_DIR/"
  mv "./${SERVICE_NAME}-key.pem" "$DEST_DIR/"

  echo "--> Success! Certificate for '$SERVICE_NAME' created in $DEST_DIR"
  echo ""
done

echo "=================================================="
echo "   ALL CERTIFICATES WERE CREATED SUCCESSFULLY!   "
echo "=================================================="

K8S_CERTS_DIR="./infra/k8s/manifests/k8s_certs.yaml"

echo ""
echo "--> Creating Kubernetes Secrets..."
echo "--------------------------------------------------"
echo "---" >"${K8S_CERTS_DIR}"
kubectl create secret generic steering-server-certs --from-file=steering-server.pem=./certs/steering-service/steering-server.pem --from-file=steering-server-key.pem=./certs/steering-service/steering-server-key.pem --dry-run=client -o yaml >>"${K8S_CERTS_DIR}"
echo "---" >>"${K8S_CERTS_DIR}"
kubectl create secret generic delivery-node-1-certs --from-file=delivery-node.pem=./certs/delivery-nodes/delivery-node-1.pem --from-file=delivery-node-key.pem=./certs/delivery-nodes/delivery-node-1-key.pem --dry-run=client -o yaml >>"${K8S_CERTS_DIR}"
echo "---" >>"${K8S_CERTS_DIR}"
kubectl create secret generic delivery-node-2-certs --from-file=delivery-node.pem=./certs/delivery-nodes/delivery-node-2.pem --from-file=delivery-node-key.pem=./certs/delivery-nodes/delivery-node-2-key.pem --dry-run=client -o yaml >>"${K8S_CERTS_DIR}"
echo "---" >>"${K8S_CERTS_DIR}"
kubectl create secret generic delivery-node-3-certs --from-file=delivery-node.pem=./certs/delivery-nodes/delivery-node-3.pem --from-file=delivery-node-key.pem=./certs/delivery-nodes/delivery-node-3-key.pem --dry-run=client -o yaml >>"${K8S_CERTS_DIR}"
