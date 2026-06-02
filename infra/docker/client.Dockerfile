FROM nginx:alpine

COPY infra/k8s/client-nginx.conf /etc/nginx/nginx.conf

COPY client/assets /usr/share/nginx/html/assets
COPY client/index.html /usr/share/nginx/html/

RUN chmod -R 755 /usr/share/nginx/html/

EXPOSE 80
