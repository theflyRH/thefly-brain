# The Fly brain: Brian2 whole-brain LIF of FlyWire v783, websocket server.
# gcc is present in the full python image so Brian2 can use its Cython backend.
FROM python:3.12

WORKDIR /app
RUN pip install --no-cache-dir brian2 numpy pandas pyarrow websockets cython setuptools

# Connectome data (Shiu et al. 2024 repository, MIT): completeness + connectivity v783
RUN mkdir -p data && \
    curl -fsSL -o data/Completeness_783.csv https://raw.githubusercontent.com/philshiu/Drosophila_brain_model/main/Completeness_783.csv && \
    curl -fsSL -o data/Connectivity_783.parquet https://raw.githubusercontent.com/philshiu/Drosophila_brain_model/main/Connectivity_783.parquet && \
    ls -la data

COPY flybrain ./flybrain

ENV FLYBRAIN_DATA=/app/data \
    PYTHONUNBUFFERED=1
EXPOSE 8765
CMD ["python", "-m", "flybrain.server", "--host", "0.0.0.0", "--port", "8765"]
