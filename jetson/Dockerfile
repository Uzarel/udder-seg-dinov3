# Use your Jetson/Orin-compatible ML base image (JetPack 5/6).
FROM dustynv/l4t-ml:r35.4.1

ENV HOST=0.0.0.0 \
    PORT=8000 \
    MODEL_PATH=/models/convnext_tiny_pruned_reshaped.onnx \
    WARMUP_H=480 \
    WARMUP_W=640 \
    WARMUP_RUNS=5

RUN python3 -m pip install --no-cache-dir --upgrade pip
RUN python3 -m pip install --no-cache-dir fastapi "uvicorn[standard]" python-multipart

WORKDIR /app
COPY app /app/app
EXPOSE 8000

CMD ["python3", "-m", "uvicorn", "app.serve_cuda:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
