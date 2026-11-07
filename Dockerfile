# ---- build the React UI ----
FROM node:22-alpine AS web
WORKDIR /web
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# ---- Python API that also serves the UI ----
FROM python:3.12-slim
WORKDIR /app/backend
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
# CPU-only torch keeps the image small (sentence-transformers needs torch for BGE).
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
COPY backend/requirements.txt .
RUN pip install -r requirements.txt
COPY backend/ ./
COPY --from=web /web/dist /app/frontend/dist
# Hugging Face Spaces expects port 7860; other hosts can override with $PORT.
ENV PORT=7860 HF_HOME=/tmp/hf CHROMA_DIR=/tmp/chroma
EXPOSE 7860
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT}
