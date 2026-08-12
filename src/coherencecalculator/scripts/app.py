#!/usr/bin/env python3
"""
CCC FastAPI server (uvicorn-ready) with single + batch inference.

Input:
- POST /infer accepts either:
  1) a single object {"filename": "...", "text": "..."}
  2) a list of such objects [{"filename": "...", "text": "..."}, ...]
  3) an object {"items": [ ... ]}

Internally:
- Convert JSON -> pandas.DataFrame
- Compute features using:
    ts_df = timeseries(vecLoader=state.models['vecs'], inputDf=df, fileCol='filename', textCol='text')
    agg_df = agg(vecLoader=state.models['vecs'], inputTimeseries=ts_df)

Notes:
- Uses an asyncio.Semaphore to cap concurrent inferences in-process.
- Uses a threading.Lock to serialize model access (keep if components are not thread-safe).
"""

from __future__ import annotations

import os
import sys
import time
import json
import asyncio
import threading
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from coherencecalculator.pipelines.timeseries import timeseries
from coherencecalculator.pipelines.agg import agg
from coherencecalculator.tools.vecloader import VecLoader

import torch

# ------------------ Metadata for /meta endpoints ------------------ #
META_DATA = [
    {"server_name": "CCC FastAPI Server"},
    {"server_version": "2.0"},
    {"server_type": "text"},
]

# You can import/reuse your FEATURE_DATA list as-is; kept compact here.
FEATURE_DATA = [
    {'wordCoherenceSeq': 'Cosine similarity between consecutive words. FastText embeddings.'},
    {'wordCoherenceStaticCentroid': 'Cosine similarity between each word and a fixed centroid vector. FastText embeddings.'},
    {'wordCoherenceCumulativeCentroid': 'Cosine similarity between each word and a moving average vector. FastText embeddings.'},
    {'phraseCoherenceSeq': 'Cosine similarity between consecutive noun phrases. FastText embeddings.'},
    {'phraseCoherenceStaticCentroid': 'Cosine similarity between each noun phrase and a fixed centroid vector. FastText embeddings.'},
    {'phraseCoherenceCumulativeCentroid': 'Cosine similarity between each noun phrase and a moving average vector. FastText embeddings.'},
    {'sentCoherenceSeq': 'Cosine similarity between consecutive sentences. FastText embeddings.'},
    {'sentCoherenceStaticCentroid': 'Cosine similarity between each sentence and a fixed centroid vector. FastText embeddings.'},
    {'sentCoherenceCumulativeCentroid': 'Cosine similarity between each sentence and a moving average vector. FastText embeddings.'},
    {'sentCoherenceWeightedSeq': 'Cosine similarity with IDF weights.'},
    {'sentCoherenceWeightedStaticCentroid': 'Cosine similarity with IDF weights to a static centroid.'},
    {'sentCoherenceWeightedCumulativeCentroid': 'Cosine similarity with IDF weights to a moving centroid.'},
    {'wordCoherenceBertSumSeq': 'Cosine similarity between consecutive BERT tokens (sum of last 4 layers).'},
    {'sentCoherenceSentBertSeq': 'Cosine similarity between consecutive sentences (Sentence-BERT).'},
    {'sentCoherenceSimCSESeq': 'Cosine similarity (SimCSE).'},
    {'sentCoherenceDiffCSESeq': 'Cosine similarity (DiffCSE).'},
    {'avg_ppl': 'Average perplexity of the full transcript.'},
    {'sliding_window': 'Sliding window perplexities (e.g., Pythia 1.4B, window 64).'},
    {'sliding_window_batch': 'Sliding window perplexities with batch averages.'},
    {'contextmodel': 'Sentence-level perplexity with previous context.'},
    {'topicmodel': 'Sentence-level perplexity with summary.'},
    {'number_of_nodes': 'Speechgraph nodes.'},
    {'number_of_edges': 'Speechgraph edges.'},
    {'PE': 'Parallel edges (speechgraph).'},
    {'number_scc': 'Strongly connected components (speechgraph).'},
    {'LSC': 'Largest strongly connected component.'},
    {'density': 'Speechgraph density.'},
    {'degree_average': 'Average degree (speechgraph).'},
    {'degree_std': 'Std dev of degree (speechgraph).'},
    {'L1': 'Loop of length 1 (speechgraph).'},
]


# ------------------ Concurrency + thread safety ------------------ #
DEFAULT_MAX_CONCURRENT = int(os.environ.get("CCC_MAX_CONCURRENT", "1"))
INFERENCE_SEMAPHORE = asyncio.Semaphore(DEFAULT_MAX_CONCURRENT)

# If you know your stack is thread-safe, you *can* remove this later.
INFERENCE_LOCK = threading.Lock()


# ------------------ State ------------------ #
@dataclass
class State:
    models: Dict[str, Any] = field(default_factory=dict)
    ready: asyncio.Event = field(default_factory=asyncio.Event)
    device: Optional[torch.device] = None
    device_idx: int = -1
    device_str: str = "cpu"


state = State()


def _load_models_sync(state: State) -> None:
    """Synchronous model/resource loading (run in a thread at startup)."""
    use_cuda = torch.cuda.is_available()
    state.device = torch.device("cuda", 0) if use_cuda else torch.device("cpu")
    state.device_str = "cuda" if use_cuda else "cpu"
    state.device_idx = 0 if use_cuda else -1

    print(f"[{time.strftime('%X')}] Starting model loading...", flush=True)
    print(f"[{time.strftime('%X')}] Using device: {state.device_str}", flush=True)

    print(f"[{time.strftime('%X')}] Loading VecLoader...", flush=True)
    state.models["vecs"] = VecLoader(device=state.device_str)
    print(f"[{time.strftime('%X')}] VecLoader loaded successfully.", flush=True)


def _df_to_response_rows(agg_df: pd.DataFrame, filename_col: str = "filename") -> List[Dict[str, Any]]:
    """Return per-row JSON-serializable dicts."""
    if not isinstance(agg_df, pd.DataFrame) or agg_df.empty:
        return []

    out_rows: List[Dict[str, Any]] = []
    for _, row in agg_df.iterrows():
        row_dict: Dict[str, Any] = {}
        for col, v in row.items():
            if isinstance(v, np.generic):
                v = v.item()
            row_dict[col] = v
        # Ensure filename is present and named consistently
        if "file" in row_dict and filename_col not in row_dict:
            row_dict[filename_col] = row_dict.pop("file")
        out_rows.append(row_dict)
    return out_rows


def _compute_features(input_df: pd.DataFrame, state: State) -> pd.DataFrame:
    """
    Compute CCC features for a DataFrame with columns:
      - filename
      - text
    Returns an aggregated feature DataFrame (one row per input).
    """
    required = {"filename", "text"}
    missing = required - set(input_df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # normalize types
    df = input_df.copy()
    df["filename"] = df["filename"].astype(str)
    df["text"] = df["text"].astype(str)

    # Drop rows with empty text to avoid downstream crashes
    df = df[df["text"].str.strip().astype(bool)]
    if df.empty:
        return pd.DataFrame()

    with INFERENCE_LOCK:
        ts_df = timeseries(
            vecLoader=state.models["vecs"],
            inputDf=df,
            fileCol="filename",
            textCol="text",
        )
        agg_df = agg(
            vecLoader=state.models["vecs"],
            inputTimeseries=ts_df,
        )
    return agg_df


# ------------------ API models ------------------ #
class Item(BaseModel):
    filename: str = Field(..., description="Unique id / filename for the transcript")
    text: str = Field(..., description="Transcript text")


class Batch(BaseModel):
    items: List[Item]


# ------------------ FastAPI app ------------------ #
app = FastAPI(title="CCC API", version="2.0")


@app.on_event("startup")
async def _startup() -> None:
    # Load models in a background thread so startup doesn't block the event loop.
    loop = asyncio.get_running_loop()
    try:
        await loop.run_in_executor(None, _load_models_sync, state)
        state.ready.set()
        print(f"[{time.strftime('%X')}] All models loaded and ready.", flush=True)
    except Exception as e:
        print(f"[{time.strftime('%X')}] ERROR during model loading: {e}", flush=True)
        traceback.print_exc()
        # do not set ready


@app.get("/health")
async def health() -> JSONResponse:
    if state.ready.is_set():
        return JSONResponse({"status": "ok"})
    return JSONResponse({"status": "loading"}, status_code=503)


@app.get("/meta/server")
async def meta_server() -> List[Dict[str, str]]:
    return META_DATA


@app.get("/meta/feature")
async def meta_feature() -> List[Dict[str, str]]:
    return FEATURE_DATA


def _payload_to_df(payload: Any) -> pd.DataFrame:
    """
    Accepts:
      - dict with filename/text
      - list of dicts
      - dict with items: [...]
    Returns DataFrame with columns filename, text
    """
    if isinstance(payload, dict) and "items" in payload:
        payload = payload["items"]

    if isinstance(payload, dict):
        rows = [payload]
    elif isinstance(payload, list):
        rows = payload
    else:
        raise ValueError("Body must be an object, a list of objects, or {'items': [...]}")

    # Minimal validation to produce good errors
    for i, r in enumerate(rows):
        if not isinstance(r, dict):
            raise ValueError(f"Item {i} is not an object")
        if "filename" not in r or "text" not in r:
            raise ValueError(f"Item {i} must contain 'filename' and 'text'")

    return pd.DataFrame(rows, columns=["filename", "text"])


@app.post("/infer")
async def infer(request: Request) -> JSONResponse:
    if not state.ready.is_set():
        raise HTTPException(status_code=503, detail="Model not ready")

    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Request body must be JSON")

    try:
        input_df = _payload_to_df(payload)
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))

    async with INFERENCE_SEMAPHORE:
        try:
            agg_df = await asyncio.get_running_loop().run_in_executor(None, _compute_features, input_df, state)
        except Exception as e:
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Feature extraction failed: {e}")

    if not isinstance(agg_df, pd.DataFrame) or agg_df.empty:
        return JSONResponse({"n": 0, "rows": []})

    # Drop heavy cols if present
    for col in ("text",):
        if col in agg_df.columns:
            agg_df = agg_df.drop(columns=[col])
    agg_df = agg_df.replace([np.nan, np.inf, -np.inf], None)
    rows = _df_to_response_rows(agg_df, filename_col="filename")
    return JSONResponse({"n": len(rows), "rows": rows})


# Optional: a root route
@app.get("/")
async def root() -> Dict[str, Any]:
    return {"service": "ccc", "endpoints": ["/health", "/meta/server", "/meta/feature", "/infer"]}
