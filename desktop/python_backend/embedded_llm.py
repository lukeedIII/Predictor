"""
Embedded Local LLM — Qwen2.5-1.5B-Instruct
============================================
Self-contained fallback LLM that runs entirely on the local machine.
Downloads the model on first use (~3GB) and caches it in HuggingFace cache.
Loads lazily — zero overhead until actually called.

Used as the LAST fallback when OpenAI, Gemini, and Ollama are all unavailable.
"""

import logging
import os
import threading
from typing import Optional, List, Dict

_model = None
_tokenizer = None
_lock = threading.Lock()
_load_error: Optional[str] = None
_is_loading = False

# Model to download — Qwen2.5-0.5B-Instruct
# Tiny (~1GB), Apache 2.0 license, no HF auth needed
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
MODEL_LABEL = "qwen2.5-0.5b"


def is_available() -> bool:
    """Check if the model is loaded or can be loaded."""
    if _model is not None:
        return True
    if _load_error:
        return False
    # Check if transformers is importable
    try:
        import transformers  # noqa: F401
        return True
    except ImportError:
        return False


def is_loaded() -> bool:
    """Check if model is currently loaded in memory."""
    return _model is not None


def get_status() -> dict:
    """Return status info for the UI."""
    return {
        "model_id": MODEL_ID,
        "label": MODEL_LABEL,
        "loaded": _model is not None,
        "loading": _is_loading,
        "error": _load_error,
        "available": is_available(),
    }


def _load_model():
    """Load the model into memory. Thread-safe, loads only once."""
    global _model, _tokenizer, _load_error, _is_loading

    if _model is not None:
        return True

    with _lock:
        # Double-check after acquiring lock
        if _model is not None:
            return True

        _is_loading = True
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            logging.info(f"[EmbeddedLLM] Loading {MODEL_ID}... (first run downloads ~3GB)")

            # Determine device — prefer GPU if VRAM available, else CPU
            device = "cpu"
            dtype = torch.float32
            if torch.cuda.is_available():
                try:
                    free_vram = torch.cuda.mem_get_info(0)[0] / 1e9
                    if free_vram > 3.5:
                        device = "cuda"
                        dtype = torch.float16
                        logging.info(f"[EmbeddedLLM] Using GPU ({free_vram:.1f}GB free)")
                    else:
                        logging.info(f"[EmbeddedLLM] GPU VRAM too low ({free_vram:.1f}GB), using CPU")
                except Exception:
                    pass

            _tokenizer = AutoTokenizer.from_pretrained(
                MODEL_ID,
                trust_remote_code=True,
            )

            _model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID,
                torch_dtype=dtype,
                device_map=device if device == "cuda" else None,
                trust_remote_code=True,
            )

            if device == "cpu":
                _model = _model.to("cpu")

            _model.eval()
            _load_error = None
            _is_loading = False
            logging.info(f"[EmbeddedLLM] {MODEL_ID} loaded on {device} ✅")
            return True

        except Exception as e:
            _load_error = str(e)[:200]
            _is_loading = False
            logging.error(f"[EmbeddedLLM] Failed to load: {_load_error}")
            return False


def generate(system_prompt: str, user_message: str,
             conversation_history: Optional[List[Dict]] = None,
             max_new_tokens: int = 1024,
             temperature: float = 0.7) -> str:
    """
    Generate a response using the embedded model.
    Loads the model on first call (blocking, ~30-60s first time).
    """
    if not _load_model():
        raise RuntimeError(f"Embedded LLM failed to load: {_load_error}")

    import torch

    # Build chat messages in the format Qwen expects
    messages = [{"role": "system", "content": system_prompt}]

    if conversation_history:
        for msg in conversation_history[-6:]:  # Fewer turns for small model
            role = "user" if msg.get("role") == "user" else "assistant"
            content = msg.get("content", "")[:300]  # Shorter context for small model
            messages.append({"role": role, "content": content})

    messages.append({"role": "user", "content": user_message})

    # Use the chat template
    text = _tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = _tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"]

    # Move to same device as model
    device = next(_model.parameters()).device
    input_ids = input_ids.to(device)

    # Truncate if too long (1.5B model context is ~32K but we keep it short)
    max_input = 4096
    if input_ids.shape[1] > max_input:
        input_ids = input_ids[:, -max_input:]

    with torch.no_grad():
        output = _model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.9,
            do_sample=True,
            pad_token_id=_tokenizer.pad_token_id or _tokenizer.eos_token_id,
        )

    # Decode only the generated tokens (skip input)
    generated_ids = output[0][input_ids.shape[1]:]
    reply = _tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    if not reply:
        reply = "I couldn't generate a response. Please try again."

    return reply


def preload_async():
    """Start loading the model in a background thread (non-blocking)."""
    if _model is not None or _is_loading:
        return
    t = threading.Thread(target=_load_model, daemon=True)
    t.start()
    logging.info("[EmbeddedLLM] Background preload started")


def unload():
    """Unload the model to free memory."""
    global _model, _tokenizer
    with _lock:
        if _model is not None:
            del _model
            del _tokenizer
            _model = None
            _tokenizer = None
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            logging.info("[EmbeddedLLM] Model unloaded")
