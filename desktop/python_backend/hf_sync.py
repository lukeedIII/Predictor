"""
hf_sync.py — Hugging Face Hub Model Sync Utility
===============================================
Handles uploading and downloading trained models to/from Hugging Face.
Requires HUGGINGFACE_TOKEN and HF_REPO_ID in .env or os.environ.
"""

import os
import shutil
import logging
import config
from huggingface_hub import HfApi, HfFileSystem, snapshot_download

logger = logging.getLogger(__name__)

def get_config():
    """Retrieve HF credentials from environment."""
    token = os.environ.get("HUGGINGFACE_TOKEN", "").strip()
    repo_id = os.environ.get("HF_REPO_ID", "").strip()
    return token, repo_id

def push_to_hub():
    """Upload all files in config.MODEL_DIR to the HF repo."""
    token, repo_id = get_config()
    if not token or not repo_id:
        return {"success": False, "error": "Missing HUGGINGFACE_TOKEN or HF_REPO_ID"}

    api = HfApi(token=token)
    
    try:
        # Create repo if it doesn't exist
        api.create_repo(repo_id=repo_id, exist_ok=True, private=True)
        
        # Upload folder
        logger.info(f"Pushing models from {config.MODEL_DIR} to {repo_id}...")
        api.upload_folder(
            folder_path=config.MODEL_DIR,
            repo_id=repo_id,
            commit_message="Update trained models from Nexus Shadow-Quant",
            repo_type="model"
        )
        return {"success": True, "repo": repo_id}
    except Exception as e:
        logger.error(f"HF Push failed: {e}")
        return {"success": False, "error": str(e)}

def pull_from_hub():
    """Download latest models from HF to config.MODEL_DIR."""
    token, repo_id = get_config()
    if not token or not repo_id:
        return {"success": False, "error": "Missing HUGGINGFACE_TOKEN or HF_REPO_ID"}

    try:
        logger.info(f"Pulling models from {repo_id} to {config.MODEL_DIR}...")
        
        # Download everything from the repo to a temporary location
        # Then move to MODEL_DIR
        downloaded_path = snapshot_download(
            repo_id=repo_id,
            token=token,
            local_dir=config.MODEL_DIR,
            local_dir_use_symlinks=False,
            repo_type="model",
            ignore_patterns=[".git*", "*.md"]
        )
        
        return {"success": True, "path": downloaded_path}
    except Exception as e:
        logger.error(f"HF Pull failed: {e}")
        return {"success": False, "error": str(e)}

def has_models():
    """Check if we have local model files already."""
    model_files = ["predictor_v3.joblib", "nexus_lstm_v3.pth", "feature_scaler_v3.pkl"]
    return any(os.path.exists(os.path.join(config.MODEL_DIR, f)) for f in model_files)
