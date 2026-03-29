import json
import pickle
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

SUPPORTED_TOKENIZATION_METHODS = ("standard", "remi", "cp_limb_v1")
TOKENIZATION_METADATA_FILENAME = "tokenization_metadata.json"


def supported_modes_for_display() -> str:
    ordered = sorted(SUPPORTED_TOKENIZATION_METHODS)
    return ", ".join(ordered)


def validate_tokenization_method(tokenization_method: Any, context: str = "tokenization_method") -> str:
    if not isinstance(tokenization_method, str):
        raise ValueError(
            f"Invalid {context}: {tokenization_method!r}. "
            f"Supported: {supported_modes_for_display()}"
        )

    normalized = tokenization_method.strip().lower()
    if not normalized:
        raise ValueError(
            f"Invalid {context}: empty value. "
            f"Supported: {supported_modes_for_display()}"
        )

    if normalized not in SUPPORTED_TOKENIZATION_METHODS:
        raise ValueError(
            f"Unsupported {context}: {tokenization_method}. "
            f"Supported: {supported_modes_for_display()}"
        )

    return normalized


def write_dataset_tokenization_metadata(
    data_dir: str,
    tokenization_method: str,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> str:
    mode = validate_tokenization_method(tokenization_method, context="dataset tokenization_method")

    out_dir = Path(data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / TOKENIZATION_METADATA_FILENAME

    payload = {
        "tokenization_method": mode,
        "schema_version": 1,
    }

    if extra_metadata:
        for key, value in extra_metadata.items():
            if key not in payload:
                payload[key] = value

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return str(meta_path)


def read_dataset_tokenization_metadata(data_dir: Optional[str]) -> Optional[dict]:
    if not data_dir:
        return None

    meta_path = Path(data_dir) / TOKENIZATION_METADATA_FILENAME
    if not meta_path.exists():
        return None

    with open(meta_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    mode = validate_tokenization_method(
        payload.get("tokenization_method"),
        context=f"{TOKENIZATION_METADATA_FILENAME}.tokenization_method",
    )
    payload["tokenization_method"] = mode
    return payload


def infer_vocab_tokenization_method(vocab_payload: Any) -> Optional[str]:
    if isinstance(vocab_payload, dict):
        if "tokenization_method" in vocab_payload:
            return validate_tokenization_method(
                vocab_payload.get("tokenization_method"),
                context="vocab tokenization_method",
            )
        if vocab_payload.get("format") == "cp_limb_v1":
            return "cp_limb_v1"
        return None

    if isinstance(vocab_payload, tuple) and len(vocab_payload) >= 2:
        # Legacy tuple payload is shared by standard/remi and cannot be disambiguated.
        return None

    return None


def resolve_artifact_tokenization_method(vocab_path: str, data_dir: Optional[str] = None) -> Tuple[Optional[str], str]:
    vocab_mode: Optional[str] = None
    if vocab_path:
        with open(vocab_path, "rb") as f:
            vocab_payload = pickle.load(f)
        vocab_mode = infer_vocab_tokenization_method(vocab_payload)

    dataset_mode: Optional[str] = None
    metadata = read_dataset_tokenization_metadata(data_dir)
    if metadata is not None:
        dataset_mode = metadata["tokenization_method"]

    if vocab_mode and dataset_mode and vocab_mode != dataset_mode:
        raise ValueError(
            "Artifact tokenization metadata mismatch: "
            f"vocab={vocab_mode}, dataset={dataset_mode}."
        )

    if dataset_mode:
        return dataset_mode, "dataset_metadata"
    if vocab_mode:
        return vocab_mode, "vocab"
    return None, "unknown"


def ensure_runtime_mode_compatibility(
    requested_mode: str,
    vocab_path: str,
    data_dir: Optional[str] = None,
    allow_legacy_standard: bool = True,
) -> str:
    mode = validate_tokenization_method(requested_mode, context="data.tokenization_method")
    artifact_mode, artifact_source = resolve_artifact_tokenization_method(vocab_path, data_dir)

    if artifact_mode is None:
        if allow_legacy_standard and mode == "standard":
            return mode
        raise ValueError(
            "Cannot verify artifact tokenization mode from vocab/dataset metadata. "
            "Legacy tuple vocab is ambiguous between standard and remi. "
            f"Requested mode: {mode}."
        )

    if mode != artifact_mode:
        raise ValueError(
            "Tokenization mode mismatch: "
            f"requested={mode}, artifact={artifact_mode} ({artifact_source})."
        )

    return mode
