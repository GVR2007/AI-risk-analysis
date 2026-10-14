"""Pre-download every selectable algorithm's .pkl from Hugging Face into models/, at deploy
build time rather than on a live user request.

Without this, the first visitor to pick a non-default algorithm (e.g. the 213MB Random Forest
Regressor) would trigger that download inside the request itself — on a single-worker gunicorn
process this can exceed the request timeout and take the whole site down until it restarts.
Running this once during the Render build means every file is already on disk before the app
ever serves traffic, so _resolve_model_path() in risk_models.py always finds it locally.

Run manually with: python app/prefetch_models.py
"""
from __future__ import annotations

import time

from risk_models import (
    CLASSIFICATION_ALGORITHMS,
    CLUSTERING_ALGORITHMS,
    MODELS,
    REGRESSION_ALGORITHMS,
    STORE,
    _resolve_model_path,
)

ALL_FILENAMES = sorted(set(REGRESSION_ALGORITHMS.values())
                        | set(CLASSIFICATION_ALGORITHMS.values())
                        | set(CLUSTERING_ALGORITHMS.values()))


def main():
    print(f"Checking {len(ALL_FILENAMES)} algorithm files (deployed defaults are already in "
          f"{STORE.relative_to(STORE.parent.parent)}, the rest come from Hugging Face)...")
    for filename in ALL_FILENAMES:
        already_local = (STORE / filename).exists() or (MODELS / filename).exists()
        start = time.time()
        path = _resolve_model_path(filename)
        tag = "already present" if already_local else f"downloaded in {time.time() - start:.1f}s"
        size_mb = path.stat().st_size / 1_000_000
        print(f"  {filename:<48s} {size_mb:7.1f} MB  ({tag})")
    print("All algorithm files are present on disk.")


if __name__ == "__main__":
    main()
