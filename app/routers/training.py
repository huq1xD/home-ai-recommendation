"""
training.py
Endpoint để trigger CF model training thủ công (cho demo capstone).
"""
import asyncio
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException
from app.services.behavior_service import load_behavior_matrix
import app.services.cf_model as cf_model
from app.services.cf_model import train_model, load_model

router = APIRouter()

_train_lock = asyncio.Lock()
_train_status = "idle"
_train_started_at = None
_train_finished_at = None
_train_error = None


def _utc_now_iso():
    return datetime.now(timezone.utc).isoformat()


async def _run_training_job():
    global _train_status, _train_started_at, _train_finished_at, _train_error

    async with _train_lock:
        _train_status = "running"
        _train_started_at = _utc_now_iso()
        _train_finished_at = None
        _train_error = None

    try:
        matrix, user_index, item_index = await load_behavior_matrix()

        if matrix is None or matrix.nnz == 0:
            async with _train_lock:
                _train_status = "skipped"
                _train_finished_at = _utc_now_iso()
            return

        await asyncio.to_thread(train_model, matrix, user_index, item_index)
        load_model()

        async with _train_lock:
            _train_status = "success"
            _train_finished_at = _utc_now_iso()
    except Exception as exc:
        async with _train_lock:
            _train_status = "failed"
            _train_error = str(exc)
            _train_finished_at = _utc_now_iso()

@router.post("/train", status_code=202)
async def trigger_training():
    """
    Trigger CF model training thủ công.
    Trả về ngay 202 và chạy train trong nền.
    """
    global _train_status, _train_started_at, _train_finished_at, _train_error

    if _train_status in {"queued", "running"}:
        raise HTTPException(
            status_code=409,
            detail="Training is already running",
        )

    _train_status = "queued"
    _train_started_at = _utc_now_iso()
    _train_finished_at = None
    _train_error = None

    asyncio.create_task(_run_training_job())
    return {
        "status": "accepted",
        "message": "Training job queued",
    }


@router.get("/model-status")
def model_status():
    """Kiểm tra trạng thái model hiện tại."""
    import os
    from app.services.cf_model import MODEL_PATH, METADATA_PATH

    meta = None
    try:
        if os.path.exists(METADATA_PATH):
            import json
            with open(METADATA_PATH, "r", encoding="utf-8") as f:
                meta = json.load(f)
    except Exception:
        meta = None

    return {
        "training_status": _train_status,
        "training_started_at": _train_started_at,
        "training_finished_at": _train_finished_at,
        "training_error": _train_error,
        "model_loaded": cf_model._model is not None,
        "model_file_exists": os.path.exists(MODEL_PATH),
        "num_users": len(cf_model._user_index),
        "num_items": len(cf_model._item_index),
        "last_saved_metadata": meta,
    }


@router.post("/reload-model", status_code=200)
def reload_model_endpoint():
    """
    Reload CF model from disk into memory without restarting server.
    Useful after training completes or if model gets corrupted.
    """
    try:
        success = load_model()
        return {
            "success": success,
            "model_loaded": cf_model._model is not None,
            "num_users": len(cf_model._user_index),
            "num_items": len(cf_model._item_index),
            "message": "Model reloaded successfully" if success else "Failed to load model",
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"Error reloading model: {e}",
            "error": str(e),
        }


@router.post("/retrain-force")
async def force_retrain():
    """
    Delete old model files and force re-train from scratch.
    Fixes index/model mismatch issues.
    """
    import os
    from app.services.cf_model import MODEL_PATH, INDEX_PATH, METADATA_PATH
    
    # Delete old files
    for fpath in [MODEL_PATH, INDEX_PATH, METADATA_PATH]:
        if os.path.exists(fpath):
            try:
                os.remove(fpath)
                print(f"[CF] Deleted {fpath}")
            except Exception as e:
                return {"success": False, "message": f"Failed to delete {fpath}: {e}"}
    
    # Clear globals
    cf_model._model = None
    cf_model._user_index = {}
    cf_model._item_index = {}
    cf_model._item_index_reverse = {}
    
    # Trigger training
    asyncio.create_task(_run_training_job())
    return {
        "status": "accepted",
        "message": "Old model deleted. Training from scratch queued.",
    }
