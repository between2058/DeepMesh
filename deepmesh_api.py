"""
DeepMesh FastAPI Service

Single-file API that converts 3D meshes / point clouds into high-quality
artist-grade meshes via auto-regressive transformer generation.

Model: DeepMesh 551M (Hourglass Transformer + Michelangelo point-cloud encoder)
Paper: https://arxiv.org/abs/2503.15265
"""

import os
import shutil
import uuid
import gc
import asyncio
import datetime
import logging
import logging.handlers
import tempfile

import torch
import trimesh
import numpy as np

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from typing import Optional, Literal
from pydantic import BaseModel, Field

# =============================================================================
# Logging
# =============================================================================

os.makedirs("/app/logs", exist_ok=True)


class TaiwanFormatter(logging.Formatter):
    _TZ = datetime.timezone(datetime.timedelta(hours=8))

    def formatTime(self, record, datefmt=None):
        dt = datetime.datetime.fromtimestamp(record.created, tz=self._TZ)
        if datefmt:
            return dt.strftime(datefmt)
        return dt.strftime("%Y-%m-%d %H:%M:%S") + f",{record.msecs:03.0f}"


class HealthCheckFilter(logging.Filter):
    def filter(self, record):
        return "GET /health" not in record.getMessage()


def _rotating_file_handler(filename: str, formatter: logging.Formatter) -> logging.Handler:
    handler = logging.handlers.TimedRotatingFileHandler(
        f"/app/logs/{filename}",
        when="midnight",
        interval=1,
        backupCount=14,
        encoding="utf-8",
    )
    handler.setFormatter(formatter)
    return handler


_fmt = TaiwanFormatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
_access_fmt = TaiwanFormatter("%(asctime)s %(message)s")

logger = logging.getLogger("app")
logger.setLevel(logging.DEBUG)
logger.propagate = False
logger.addHandler(_rotating_file_handler("app.log", _fmt))
logger.addHandler(logging.StreamHandler())

_uvicorn_access = logging.getLogger("uvicorn.access")
_uvicorn_access.addFilter(HealthCheckFilter())
_uvicorn_access.addHandler(_rotating_file_handler("access.log", _access_fmt))

_uvicorn = logging.getLogger("uvicorn")
_uvicorn.addHandler(_rotating_file_handler("uvicorn.log", _fmt))

# =============================================================================
# FastAPI App
# =============================================================================

app = FastAPI(title="DeepMesh API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/app/outputs")
MODEL_PATH = os.environ.get("DEEPMESH_MODEL_PATH", "/app/weights/pytorch_model.bin")
MODEL_ID = os.environ.get("DEEPMESH_MODEL_ID", "551")

os.makedirs(OUTPUT_DIR, exist_ok=True)
logger.info(f"Output directory: {OUTPUT_DIR}")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SUPPORTED_MESH_EXTENSIONS = {".obj", ".ply", ".stl", ".glb", ".gltf", ".off"}
SUPPORTED_PC_EXTENSIONS = {".npy", ".ply"}

# -- Global state --------------------------------------------------------------
model = None
gpu_lock = asyncio.Lock()


# =============================================================================
# GPU Memory Tracking
# =============================================================================

def log_gpu_memory(label: str):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024 ** 3
        reserved = torch.cuda.memory_reserved() / 1024 ** 3
        logger.info(
            f"GPU memory [{label}]: allocated={allocated:.2f} GB  reserved={reserved:.2f} GB"
        )


def flush_gpu():
    gc.collect()
    torch.cuda.empty_cache()
    log_gpu_memory("after flush")


# =============================================================================
# Error Classification
# =============================================================================

GPU_ERROR_RESPONSES = {
    503: {
        "description": "GPU OOM or model loading failed (retry after 30s)",
        "content": {
            "application/json": {
                "examples": {
                    "GPU_OOM": {
                        "value": {"detail": {"error_code": "GPU_OOM", "message": "GPU out of memory"}}
                    }
                }
            }
        },
    },
    507: {"description": "Disk full"},
    500: {"description": "Inference error"},
}


def classify_exception(e: Exception) -> tuple:
    if isinstance(e, torch.cuda.OutOfMemoryError):
        return 503, "GPU_OOM", "GPU out of memory. Retry later."
    if isinstance(e, RuntimeError) and "out of memory" in str(e).lower():
        return 503, "GPU_OOM", "GPU out of memory. Retry later."
    if isinstance(e, OSError) and getattr(e, "errno", None) == 28:
        return 507, "DISK_FULL", "Disk full. Contact administrator."
    return 500, "INFERENCE_ERROR", str(e)


# =============================================================================
# Gumbel Noise Sampling (from DeepMesh sample.py)
# =============================================================================

def add_gumbel_noise(logits, temperature):
    """As suggested by https://arxiv.org/pdf/2409.02908, use float64 for gumbel max."""
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


# =============================================================================
# Auto-regressive Sampling with KV-Cache (adapted for single GPU)
# =============================================================================

@torch.no_grad()
def ar_sample_kvcache(gpt, prompt, pc, temperature=0.5,
                      context_length=90000, window_size=9000, device="cuda"):
    """Generate mesh tokens auto-regressively with KV-cache."""
    gpt.eval()
    N = prompt.shape[0]
    end_list = [0 for _ in range(N)]

    for cur_pos in range(prompt.shape[1], context_length):
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            if cur_pos >= 9001 and (cur_pos - 9001) % 4500 == 0:
                start = 4500 + ((cur_pos - 9001) // 4500) * 4500
            else:
                start = cur_pos - 1
            input_pos = torch.arange(cur_pos, dtype=torch.long, device=device)
            prompt_input = prompt[:, start:cur_pos]
            logits = gpt(prompt_input, pc=pc, start=start,
                         window_size=window_size, input_pos=input_pos)[:, -1]
            pc = None

        logits_with_noise = add_gumbel_noise(logits, temperature)
        next_token = torch.argmax(logits_with_noise, dim=-1, keepdim=True)
        prompt = torch.cat([prompt, next_token], dim=-1)

        for u in range(N):
            if end_list[u] == 0:
                if next_token[u] == torch.tensor([4737], device=device):
                    end_list[u] = 1
        if sum(end_list) == N:
            break

    return prompt, cur_pos


# =============================================================================
# Model Loading (Lazy)
# =============================================================================

def ensure_model_loaded():
    global model
    if model is not None:
        return

    logger.info("[Lazy Load] Loading DeepMesh model...")
    log_gpu_memory("before model load")

    try:
        from lit_gpt.model_cache import GPTCache, Config
        from safetensors.torch import load_file

        model_name = f"Diff_LLaMA_{MODEL_ID}M"
        config = Config.from_name(model_name)
        logger.info(f"Model config: {model_name}")

        config.padded_vocab_size = (2 * 4 ** 3) + (8 ** 3) + (16 ** 3) + 1 + 1  # 4738
        config.block_size = 270000

        loaded_model = GPTCache(config).to(DEVICE)

        model_path = MODEL_PATH
        if model_path.endswith(".safetensors"):
            state_dict = load_file(model_path)
        else:
            state_dict = torch.load(model_path, map_location="cpu", weights_only=False)

        loaded_model.load_state_dict(state_dict, strict=False)
        loaded_model.eval()

        model = loaded_model
        log_gpu_memory("model loaded")
        logger.info("[Lazy Load] Model loaded successfully!")

    except Exception as e:
        logger.error(f"Model loading failed: {e}")
        raise RuntimeError(f"Model loading failed: {e}")


# =============================================================================
# Preprocessing Helpers
# =============================================================================

def sample_pc_from_mesh(verts, faces, pc_num=16384):
    """Sample point cloud with normals from a mesh (matching DeepMesh dataset.py)."""
    mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    points, face_idx = mesh.sample(50000, return_index=True)
    normals = mesh.face_normals[face_idx]
    # DeepMesh coordinate convention: (x,y,z) -> (z,x,y)
    pc_normal = np.concatenate(
        [points[:, [2, 0, 1]], normals[:, [2, 0, 1]]], axis=-1, dtype=np.float16
    )
    ind = np.random.choice(pc_normal.shape[0], pc_num, replace=False)
    return pc_normal[ind]


def load_mesh_input(file_path: str, point_num: int = 16384) -> np.ndarray:
    """Load a mesh file and convert to point cloud with normals."""
    mesh = trimesh.load(file_path, force="mesh")
    return sample_pc_from_mesh(mesh.vertices, mesh.faces, pc_num=point_num)


def load_pc_input(file_path: str, point_num: int = 16384) -> np.ndarray:
    """Load a point cloud file (.ply or .npy) with normals."""
    import open3d as o3d

    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".npy":
        pc_normal = np.load(file_path)
        if pc_normal.shape[1] < 6:
            raise ValueError(f"Point cloud must have 6 columns (xyz + normals), got {pc_normal.shape[1]}")
    elif ext == ".ply":
        points_raw = trimesh.load(file_path, process=False)
        if hasattr(points_raw, "vertices"):
            pts = np.asarray(points_raw.vertices)
        else:
            pts = np.asarray(points_raw)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))
        pcd.orient_normals_consistent_tangent_plane(k=30)

        points = np.asarray(pcd.points)
        normals = np.asarray(pcd.normals)
        pc_normal = np.concatenate([points, normals], axis=1)
    else:
        raise ValueError(f"Unsupported point cloud format: {ext}")

    if len(pc_normal) < point_num:
        raise ValueError(f"Point cloud has {len(pc_normal)} points, need at least {point_num}")

    if len(pc_normal) > point_num:
        indices = np.random.choice(len(pc_normal), point_num, replace=False)
        pc_normal = pc_normal[indices]

    return pc_normal.astype(np.float16)


def postprocess_tokens(output_ids: torch.Tensor) -> trimesh.Trimesh:
    """Convert generated token sequence to a mesh (matching DeepMesh sample.py)."""
    from sft.datasets.serializaiton import deserialize
    from sft.datasets.data_utils import to_mesh

    code = output_ids[1:]  # skip start token
    index = (code >= 4737).nonzero()
    if index.numel() > 0:
        code = code[:index[0, 0].item()].cpu().numpy().astype(np.int64)
    else:
        code = code.cpu().numpy().astype(np.int64)

    vertices = deserialize(code)
    if len(vertices) == 0:
        raise ValueError("Generated mesh has no vertices")

    # DeepMesh coordinate convention reversal
    vertices = vertices[..., [2, 1, 0]]

    faces = torch.arange(1, len(vertices) + 1).view(-1, 3)
    mesh = to_mesh(vertices, faces, transpose=False, post_process=True)
    return mesh


# =============================================================================
# Response Schemas
# =============================================================================

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    gpu_busy: bool
    device: str = Field(description="'cuda' or 'cpu'")


class GenerateResponse(BaseModel):
    status: str
    request_id: str
    output_urls: list[str]
    num_meshes: int


# =============================================================================
# Endpoints
# =============================================================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "gpu_busy": gpu_lock.locked(),
        "device": DEVICE,
    }


@app.post("/generate", response_model=GenerateResponse, responses=GPU_ERROR_RESPONSES)
async def generate(
    file: UploadFile = File(...),
    input_type: Literal["mesh", "pc"] = Form("mesh"),
    temperature: float = Form(0.5, ge=0.01, le=2.0, description="Sampling temperature (lower = more deterministic)"),
    max_steps: int = Form(90000, ge=1000, le=90000, description="Maximum generation steps"),
    repeat_num: int = Form(1, ge=1, le=4, description="Number of meshes to generate per input"),
    point_num: int = Form(16384, ge=4096, le=50000, description="Points to sample from input"),
    seed: int = Form(0, ge=0, description="Random seed"),
):
    # Validate file extension
    ext = os.path.splitext(file.filename or "")[1].lower()
    if input_type == "mesh" and ext not in SUPPORTED_MESH_EXTENSIONS:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported mesh format '{ext}'. Must be one of: {sorted(SUPPORTED_MESH_EXTENSIONS)}",
        )
    if input_type == "pc" and ext not in SUPPORTED_PC_EXTENSIONS:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported point cloud format '{ext}'. Must be .npy or .ply",
        )

    request_id = str(uuid.uuid4())
    req_dir = os.path.join(OUTPUT_DIR, request_id)
    os.makedirs(req_dir, exist_ok=True)

    # Save uploaded file
    input_filename = file.filename or f"input{ext}"
    input_path = os.path.join(req_dir, input_filename)
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    logger.info(f"[{request_id}] Received {input_type} file: {input_filename}")

    async with gpu_lock:
        await run_in_threadpool(ensure_model_loaded)

        def run_inference():
            try:
                torch.manual_seed(seed)
                np.random.seed(seed)

                # Load and preprocess input
                logger.info(f"[{request_id}] Preprocessing (point_num={point_num})...")
                if input_type == "mesh":
                    pc_normal = load_mesh_input(input_path, point_num=point_num)
                else:
                    pc_normal = load_pc_input(input_path, point_num=point_num)

                cond_pc = torch.tensor(pc_normal).unsqueeze(0).to(DEVICE)

                # Save input point cloud for reference
                points = cond_pc[0].cpu().numpy()
                point_cloud = trimesh.points.PointCloud(points[..., 0:3])
                point_cloud.export(os.path.join(req_dir, "input_pc.ply"))

                # Run inference
                logger.info(
                    f"[{request_id}] Running inference "
                    f"(repeat={repeat_num}, temp={temperature}, steps={max_steps})..."
                )
                log_gpu_memory("before inference")

                prompt = torch.tensor([[4736]]).to(DEVICE).repeat(repeat_num, 1)
                pc_input = cond_pc.repeat(repeat_num, 1, 1)

                output_ids, final_pos = ar_sample_kvcache(
                    model,
                    prompt=prompt,
                    pc=pc_input,
                    temperature=temperature,
                    context_length=max_steps,
                    window_size=9000,
                    device=DEVICE,
                )

                log_gpu_memory("after inference")
                logger.info(f"[{request_id}] Generation finished at step {final_pos}")

                # Post-process each generated mesh
                output_files = []
                for u in range(repeat_num):
                    try:
                        mesh = postprocess_tokens(output_ids[u])
                        filename = f"output_{u}.obj"
                        mesh.export(os.path.join(req_dir, filename))
                        output_files.append(filename)
                        logger.info(
                            f"[{request_id}] Mesh {u}: "
                            f"{len(mesh.vertices)} verts, {len(mesh.faces)} faces"
                        )
                    except Exception as e:
                        logger.warning(f"[{request_id}] Mesh {u} post-processing failed: {e}")

                if not output_files:
                    raise RuntimeError("All mesh generations failed post-processing")

                # Reset KV-cache to free memory
                model.transformer.h.pre_kv_caches = []
                model.transformer.h.post_kv_caches = []
                if hasattr(model.transformer.h, "hourglass") and model.transformer.h.hourglass is not None:
                    model.transformer.h.hourglass.pre_kv_caches = []
                    model.transformer.h.hourglass.post_kv_caches = []

                return output_files

            finally:
                flush_gpu()

        try:
            output_files = await run_in_threadpool(run_inference)
            output_urls = [f"/download/{request_id}/{f}" for f in output_files]
            return {
                "status": "ok",
                "request_id": request_id,
                "output_urls": output_urls,
                "num_meshes": len(output_files),
            }
        except Exception as e:
            logger.error(f"[{request_id}] Error: {e}")
            status_code, error_code, message = classify_exception(e)
            raise HTTPException(
                status_code=status_code,
                detail={"error_code": error_code, "message": message},
            )


@app.get("/download/{request_id}/{file_name}")
async def download_file(request_id: str, file_name: str):
    file_path = os.path.join(OUTPUT_DIR, request_id, file_name)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    media_type = "application/octet-stream"
    if file_name.endswith(".obj"):
        media_type = "model/obj"
    elif file_name.endswith(".ply"):
        media_type = "application/x-ply"

    return FileResponse(file_path, media_type=media_type, filename=file_name)


@app.on_event("shutdown")
async def cleanup():
    logger.info("Server shutting down (output files preserved in %s)", OUTPUT_DIR)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8193)
