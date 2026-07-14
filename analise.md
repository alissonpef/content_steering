## Resumo (PT-BR)
O projeto simula e avalia estratégias dinâmicas de **Content Steering** para transmissões de vídeo adaptativas via **DASH** (*Dynamic Adaptive Streaming over HTTP*), utilizando algoritmos de Aprendizado por Reforço (*Reinforcement Learning*) em um ambiente nativo de Kubernetes.

- **O que faz / Para que serve**: Ele resolve o problema de seleção dinâmica e inteligente de redes de distribuição de conteúdo (CDNs/servidores de entrega) pelos players de vídeo. O objetivo é otimizar a Qualidade de Experiência (QoE) do usuário (minimizando latência e travamentos/stalls) direcionando o player para o melhor servidor disponível.
- **Como funciona por cima**:
  1. O player de vídeo frontend no navegador (`Dash.js`) solicita uma decisão de direcionamento ao servidor de steering.
  2. O servidor de steering (`FastAPI`) lê métricas de latência do cluster Kubernetes e seleciona ordenadamente os melhores nós de entrega (`Delivery Nodes` emulados via `Caddy`) utilizando o algoritmo de aprendizado ativo (ex: *LinUCB*, *Thompson Sampling*, *PPO*, etc.).
  3. O player baixa os segmentos de vídeo do servidor recomendado e envia o feedback de latência/experiência de volta ao servidor de steering.
  4. O algoritmo atualiza seus pesos com base no feedback recebido para refinar as próximas decisões.
  5. Os dados coletados são salvos em CSV e processados em um pipeline de análise estatística e geração de gráficos de desempenho.
- **Tipo de projeto**: Trata-se de uma aplicação distribuída / simulador contendo uma API REST (backend de steering), uma interface web frontend (dashboard e player), agentes emissores de emulação de rede e pipelines de análise de logs.

## 8 Tecnologias principais
- Python (v3.12+)
- FastAPI
- Kubernetes (Kind)
- Docker
- JavaScript (Dash.js)
- Uvicorn
- NumPy
- Pandas

## Descrição para o GitHub (About)
A Kubernetes-native simulation platform to evaluate dynamic content steering in DASH streaming, built with Python, FastAPI, Kubernetes, Docker, and JavaScript. Features adaptive reinforcement learning strategies to optimize client-side CDN selection based on real-time network latency.

## Sugestão de topics do GitHub
- content-steering
- dash-streaming
- reinforcement-learning
- kubernetes
- fastapi
- docker
- network-simulation
- multi-armed-bandits
