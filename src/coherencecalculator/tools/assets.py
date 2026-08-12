"""Resolve data assets from the ``LinguisticAnomalies/Coherence`` HuggingFace repo.

The vector store, idf table and trained models/scalers are large files (on the
order of GBs), so they are no longer bundled inside the installed wheel and
looked up with ``pkg_resources.resource_filename``. Instead they are hosted on
HuggingFace Hub and downloaded on first access, then cached under
``HF_HOME``/``HF_HUB_CACHE`` by ``huggingface_hub``.
"""

from functools import lru_cache

from huggingface_hub import hf_hub_download

COHERENCE_REPO = "LinguisticAnomalies/Coherence"

# Repo-relative paths of the assets the package needs at runtime.
VECFILE = "vecs/fasttext_vectors.bin"
IDFFILE = "vecs/wikiidf_terms.csv"
MODEL = "models/model_original.pickle"
SCALER = "models/scaler_original.pickle"


@lru_cache(maxsize=None)
def get_asset(repo_path: str) -> str:
    """Return the local filesystem path of ``repo_path``, downloading it from the
    HuggingFace Coherence repo into the local hub cache on first use."""
    return hf_hub_download(repo_id=COHERENCE_REPO, filename=repo_path)
