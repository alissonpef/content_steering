FROM caddy:latest

RUN apk add --no-cache iproute2

WORKDIR /srv

EXPOSE 80
EXPOSE 443
