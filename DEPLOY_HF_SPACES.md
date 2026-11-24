# Deploy Gratuito no Hugging Face Spaces (16 GB RAM, 2 vCPUs)

Como este projeto usa **PyTorch + Embeddings Multilíngues (Transformers) + XGBoost + OCR local**, plataformas gratuitas com 512 MB de RAM (como Render Free ou Vercel) estouram memória.

O **Hugging Face Spaces (Docker SDK)** oferece **16 GB de RAM e 2 vCPUs gratuitamente**, com HTTPS automático e URL pública permanente para os instrutores e monitores do **UberHub Code Club**.

## Passo a passo (3 minutos)

1. Crie uma conta gratuita em [huggingface.co](https://huggingface.co) (não exige cartão de crédito).
2. Clique em **New Space**:
   - **Space name**: `uberhub-calibrador-questoes`
   - **SDK**: **Docker** → **Blank**
   - **Hardware**: **CPU basic · 2 vCPU · 16 GB · FREE**
   - **Visibility**: Public (ou Private se quiser restringir)
3. Suba este repositório para o Space (via `git push` ou upload direto dos arquivos `Dockerfile`, `pyproject.toml`, `README.md`, `src/`, `artifacts/`, `data/`).
4. Adicione no topo do `README.md` do Space o cabeçalho YAML abaixo (exigido pelo Hugging Face Spaces):

```yaml
---
title: Calibrador de Questoes - UberHub Code Club
emoji: 🎯
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
---
```

5. Aguarde o build do Docker (~3 minutos). Pronto! Sua aplicação estará no ar em:
   `https://<seu-usuario>-uberhub-calibrador-questoes.hf.space`
