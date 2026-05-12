"""
routers/scores.py
Endpoints to expose Collaborative Filtering scores for products.
"""
from typing import List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

import app.services.cf_model as cf_model
from app.services.cf_model import get_cf_scores
from app.services.mongo_service import get_client, DB_NAME, COLLECTION
from app.services.behavior_service import BEHAVIOR_COLLECTION

router = APIRouter()


class ScoresRequest(BaseModel):
    user_id: str
    product_ids: List[str]


class GlobalScoresResponseItem(BaseModel):
    product_id: str
    name: str
    cf_score: float


@router.get("/product/{product_id}")
async def product_score(product_id: str, user_id: str = None):
    """Return CF score for a single product for the given user."""
    if not user_id:
        raise HTTPException(status_code=400, detail="user_id is required")

    scores = get_cf_scores(user_id, [product_id])
    cf = float(scores.get(product_id, 0.0)) if scores else 0.0

    return {
        "product_id": product_id,
        "user_id": user_id,
        "cf_score": round(cf, 4),
        "model_loaded": cf_model._model is not None,
    }


@router.post("/list")
async def list_scores(req: ScoresRequest):
    """Return CF scores for a list of product IDs, sorted descending."""
    if not req.product_ids:
        raise HTTPException(status_code=400, detail="product_ids is required")

    scores = get_cf_scores(req.user_id, req.product_ids) if req.user_id else {}

    items = [
        {"product_id": pid, "cf_score": round(float(scores.get(pid, 0.0)), 4)}
        for pid in req.product_ids
    ]

    items_sorted = sorted(items, key=lambda x: x["cf_score"], reverse=True)
    return {"user_id": req.user_id, "scores": items_sorted}


@router.get("/global")
async def global_scores(user_id: str, limit: int = 100):
    """
    Return only products present in this user's behavior history,
    scored by CF and sorted from highest score to lowest score.
    """
    if not user_id:
        raise HTTPException(status_code=400, detail="user_id is required")

    db = get_client()[DB_NAME]
    col = db[COLLECTION]
    be_col = db[BEHAVIOR_COLLECTION]

    # Only score products the user has actually interacted with.
    product_ids = await be_col.distinct("productId", {"userId": user_id})
    product_ids = [str(pid) for pid in product_ids if pid]

    if not product_ids:
        return {
            "user_id": user_id,
            "total_items": 0,
            "scores": [],
        }

    docs = await col.find({"_id": {"$in": product_ids}}, {"name": 1}).to_list(length=None)
    doc_map = {str(doc.get("_id")): doc for doc in docs if doc.get("_id")}

    # Compute raw scores first for better debug visibility
    from app.services.cf_model import get_cf_raw_scores
    raw_scores = get_cf_raw_scores(user_id, product_ids)

    # Build items with raw and normalized scores
    items = []
    max_raw = max(raw_scores.values()) if raw_scores else 0.0

    for pid in product_ids:
        doc = doc_map.get(pid, {})
        raw = float(raw_scores.get(pid, 0.0)) if pid in raw_scores else 0.0
        norm = (raw / max_raw) if (max_raw > 0 and pid in raw_scores) else 0.0
        items.append({
            "product_id": pid,
            "name": doc.get("name", ""),
            "raw_score": round(raw, 6),
            "cf_score": round(norm, 6),
            "cf_covered": pid in raw_scores,
        })

    # Sort: first by cf_score desc, then by raw_score desc
    items_sorted = sorted(items, key=lambda x: (x.get("cf_score", 0), x.get("raw_score", 0)), reverse=True)
    items_limited = items_sorted[:limit] if limit and limit > 0 else items_sorted
    return {
        "user_id": user_id,
        "total_items": len(items_sorted),
        "scores": items_limited,
    }


@router.get("/popularity")
async def popularity_scores(limit: int = 100, normalize: bool = True):
    """
    Return products ranked by global popularity computed from behavior events.
    Aggregates `user_behavior_events` using weights:
      PRODUCT_VIEW=1, ADD_TO_CART=3, PURCHASE=5, RATING uses rating value.
    Query params:
      - `limit`: max number of products to return
      - `normalize`: whether to normalize scores to 0-1
    """
    db = get_client()[DB_NAME]
    be_col = db[BEHAVIOR_COLLECTION]
    prod_col = db[COLLECTION]

    pipeline = [
        {
            "$project": {
                "productId": 1,
                "score": {
                    "$switch": {
                        "branches": [
                            {"case": {"$eq": ["$eventType", "PRODUCT_VIEW"]}, "then": 1},
                            {"case": {"$eq": ["$eventType", "ADD_TO_CART"]}, "then": 3},
                            {"case": {"$eq": ["$eventType", "PURCHASE"]}, "then": 5},
                            {"case": {"$eq": ["$eventType", "RATING"]}, "then": "$rating"},
                        ],
                        "default": 1,
                    }
                }
            }
        },
        {"$group": {"_id": "$productId", "total_score": {"$sum": "$score"}, "count": {"$sum": 1}}},
        {"$sort": {"total_score": -1}},
        {"$limit": limit},
        {"$lookup": {
            "from": COLLECTION,
            "let": {"prodIdStr": "$_id"},
            "pipeline": [
                {"$match": {"$expr": {"$eq": [{"$toString": "$_id"}, "$$prodIdStr"]}}},
                {"$project": {"name": 1}}
            ],
            "as": "product"
        }},
        {"$unwind": {"path": "$product", "preserveNullAndEmptyArrays": True}},
        {"$project": {"product_id": "$_id", "total_score": 1, "count": 1, "name": {"$ifNull": ["$product.name", ""]}}},
    ]

    results = await be_col.aggregate(pipeline).to_list(length=limit)
    if not results:
        return {"total_items": 0, "scores": []}

    max_score = max(r.get("total_score", 0) for r in results) if normalize else 1.0
    items = []
    for r in results:
        score = r.get("total_score", 0)
        norm = (score / max_score) if (normalize and max_score > 0) else score
        items.append({
            "product_id": str(r.get("product_id")),
            "name": r.get("name") or "",
            "popularity_score": round(float(norm), 4) if normalize else round(float(score), 4),
            "raw_score": round(float(score), 4),
            "events_count": int(r.get("count", 0)),
        })

    return {"total_items": len(items), "scores": items}
