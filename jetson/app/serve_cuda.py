import os
import threading
import numpy as np
import cv2
import onnxruntime as ort

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import Response
from typing import Tuple

def stable_sigmoid(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    pos = x >= 0
    z = np.empty_like(x, dtype=np.float32)
    z[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    z[~pos] = ex / (1.0 + ex)
    return z

class ORTModelCUDA:
    def __init__(self, model_path: str, warm_shape: Tuple[int, int] = (480, 640), warm_runs: int = 5):
        if not os.path.exists(model_path):
            raise FileNotFoundError(model_path)

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.intra_op_num_threads = 1
        so.inter_op_num_threads = 1

        # CUDA EP only, then CPU as safety fallback (No TensorRT)
        self.sess = ort.InferenceSession(
            model_path,
            sess_options=so,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        self.inp = self.sess.get_inputs()[0]
        self.inp_name = self.inp.name

        # Warmup builds kernels
        h, w = warm_shape
        dummy = np.zeros((1, 1, h, w), dtype=np.float32)
        for _ in range(max(0, warm_runs)):
            self.sess.run(None, {self.inp_name: dummy})

    def infer_png_from_bytes(self, image_bytes: bytes, threshold: float = 0.5) -> bytes:
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError("Invalid or unsupported image payload")

        inp = (img.astype(np.float32) / 255.0)[None, None, ...]  # (1,1,H,W)
        out = self.sess.run(None, {self.inp_name: inp})[0]

        logits = np.squeeze(out).astype(np.float32)
        if logits.ndim != 2:
            logits = np.squeeze(logits)
            if logits.ndim != 2:
                raise RuntimeError(f"Unexpected output shape: {out.shape}")

        probs = stable_sigmoid(logits)
        mask = (probs >= threshold).astype(np.uint8) * 255

        ok, buf = cv2.imencode(".png", mask)
        if not ok:
            raise RuntimeError("PNG encode failed")
        return buf.tobytes()

app = FastAPI(title="Segmentation API (CUDA EP)")

_infer_lock = threading.Lock()

@app.on_event("startup")
def _startup():
    model_path = os.environ.get("MODEL_PATH", "/models/model.onnx")
    warm_h = int(os.environ.get("WARMUP_H", "480"))
    warm_w = int(os.environ.get("WARMUP_W", "640"))
    warm_runs = int(os.environ.get("WARMUP_RUNS", "5"))
    app.state.model = ORTModelCUDA(model_path, warm_shape=(warm_h, warm_w), warm_runs=warm_runs)

@app.get("/health")
def health():
    return {"ok": True, "providers": app.state.model.sess.get_providers()}

@app.post("/segment", response_class=Response, responses={200: {"content": {"image/png": {}}}})
def segment(image: UploadFile = File(...)):
    try:
        data = image.file.read()
        with _infer_lock:
            out_png = app.state.model.infer_png_from_bytes(data)
        return Response(content=out_png, media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
