"""HuggingFace Hub download and search operations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from huggingface_hub import model_info as hf_model_info
from huggingface_hub import snapshot_download


def download_model(repo_id: str, local_dir: str | None = None) -> Path:
    """Download a model from HuggingFace Hub.

    Args:
        repo_id: HuggingFace repository ID.
        local_dir: Optional directory to download into.
            If provided, files are saved there instead of the HF cache.

    Returns:
        Path to the downloaded model directory.
    """
    kwargs: dict[str, object] = {"repo_id": repo_id}
    if local_dir:
        kwargs["local_dir"] = local_dir
    try:
        path = snapshot_download(**kwargs)
        return Path(path)
    except (ValueError, OSError) as exc:
        if "fds_to_keep" in str(exc) or "bad value" in str(exc).lower():
            # Inside Textual/async runtimes, open file descriptors cause
            # multiprocessing errors. Fall back to a clean subprocess.
            return _download_in_subprocess(repo_id, local_dir)
        raise


def _download_in_subprocess(repo_id: str, local_dir: str | None) -> Path:
    """Run snapshot_download in a separate process to avoid fd issues."""
    import json
    import subprocess
    import sys

    script = (
        "import json, sys; "
        "from huggingface_hub import snapshot_download; "
        "kw = json.loads(sys.argv[1]); "
        "print(snapshot_download(**kw))"
    )
    kwargs: dict[str, object] = {"repo_id": repo_id}
    if local_dir:
        kwargs["local_dir"] = local_dir

    result = subprocess.run(
        [sys.executable, "-c", script, json.dumps(kwargs)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        error = result.stderr.strip().splitlines()
        raise RuntimeError(error[-1] if error else "download failed")

    path = result.stdout.strip().splitlines()[-1]
    return Path(path)


def get_model_info(repo_id: str) -> dict[str, Any]:
    """Get model metadata from HuggingFace Hub.

    Args:
        repo_id: HuggingFace repository ID.

    Returns:
        Dict with model metadata.
    """
    info = hf_model_info(repo_id)
    siblings = [
        {"rfilename": s.rfilename} for s in (info.siblings or [])
    ]
    return {
        "id": info.id,
        "author": info.author,
        "tags": list(info.tags or []),
        "downloads": info.downloads,
        "likes": info.likes,
        "siblings": siblings,
        "pipeline_tag": info.pipeline_tag,
        "library_name": getattr(info, "library_name", None),
    }


def search_models(
    query: str,
    mlx_only: bool = False,
    limit: int = 20,
    offset: int = 0,
) -> list[dict[str, Any]]:
    """Search HuggingFace Hub for models.

    Args:
        query: Search query string.
        mlx_only: If True, filter to MLX models only.
        limit: Maximum number of results.
        offset: Number of results to skip (for pagination).

    Returns:
        List of model info dicts.
    """
    from itertools import islice

    from huggingface_hub import list_models

    search_query = f"mlx {query}" if mlx_only else query

    results = list_models(
        search=search_query,
        sort="downloads",
        limit=offset + limit,
        expand=["safetensors"],
    )

    # Skip `offset` results, then take `limit`
    page = list(islice(results, offset, offset + limit))

    return [
        {
            "id": m.id,
            "author": m.author,
            "downloads": m.downloads,
            "likes": m.likes,
            "pipeline_tag": m.pipeline_tag,
            "tags": list(m.tags or []),
            "size_bytes": (
                m.safetensors.total if m.safetensors else None
            ),
        }
        for m in page
    ]
