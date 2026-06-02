FROM nginx:alpine
COPY infra/k8s/gateway-nginx.conf /etc/nginx/nginx.conf
EXPOSE 80
