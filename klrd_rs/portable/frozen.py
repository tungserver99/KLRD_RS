"""Portable, checksum-verified frozen E1 receipt loader."""

from dataclasses import dataclass
import hashlib
import re
from pathlib import Path
from typing import Any

import numpy as np

from ..io import digest, read_json, sha256
from ..math import key_ids
from .profile import ModelProfile


_SHA40 = re.compile(r"^[0-9a-f]{40}$")


@dataclass(frozen=True)
class FrozenE1:
    profile_id: str
    path: Path
    model_identity: dict[str, Any]
    basis: np.ndarray
    mean: np.ndarray
    P: np.ndarray
    Q: np.ndarray
    fisher: np.ndarray
    whitener: np.ndarray
    lambda_value: float
    key_bank: np.ndarray
    key_ids: list[dict]
    receipt: dict[str, Any]


def load_frozen_e1(path: str | Path, expected_profile: ModelProfile) -> FrozenE1:
    """Load only a complete, hash-consistent Llama E1 freeze."""
    expected_profile.validate()
    root = Path(path)
    receipt_path = root if root.is_file() else root / "frozen_e1.json"
    if not receipt_path.is_file():
        raise ValueError(f"MISSING_FROZEN_E1: expected {receipt_path}")
    root = receipt_path.parent.resolve()
    receipt = read_json(receipt_path)
    if receipt.get("schema") != "klrd_rs.portable.frozen_e1.v1":
        raise ValueError("FROZEN_E1_SCHEMA_MISMATCH")
    if receipt.get("profile_id") != expected_profile.profile_id or receipt.get("model_id") != expected_profile.model_id:
        raise ValueError("FROZEN_E1_PROFILE_MISMATCH")
    if expected_profile.revision is not None and receipt.get("model_revision") != expected_profile.revision:
        raise ValueError("FROZEN_E1_MODEL_REVISION_MISMATCH")
    artifacts = receipt.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise ValueError("FROZEN_E1_ARTIFACT_MANIFEST_MISSING")
    paths = {}
    for name, entry in artifacts.items():
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            raise ValueError(f"FROZEN_E1_ARTIFACT_ENTRY_INVALID: {name}")
        relative = Path(entry["path"])
        artifact_path = (root / relative).resolve()
        if relative.is_absolute() or root not in artifact_path.parents:
            raise ValueError(f"FROZEN_E1_ARTIFACT_PATH_INVALID: {name}")
        if not artifact_path.is_file():
            raise ValueError(f"FROZEN_E1_ARTIFACT_MISSING: {name}")
        actual = sha256(artifact_path)
        if actual != entry.get("sha256"):
            raise ValueError(f"FROZEN_E1_ARTIFACT_CHECKSUM_MISMATCH: {name}")
        paths[name] = artifact_path

    identity = read_json(paths["model_identity"])
    identity_unsigned = dict(identity)
    recorded_identity_sha = identity_unsigned.pop("identity_sha256", None)
    if recorded_identity_sha != digest(identity_unsigned):
        raise ValueError("FROZEN_E1_MODEL_IDENTITY_CHECKSUM_MISMATCH")
    if identity.get("profile_id") != expected_profile.profile_id or identity.get("model_id") != expected_profile.model_id:
        raise ValueError("FROZEN_E1_MODEL_IDENTITY_MISMATCH")
    if identity.get("model_type") != expected_profile.model_type or identity.get("hf_architecture") != expected_profile.hf_architecture:
        raise ValueError("FROZEN_E1_MODEL_ARCHITECTURE_MISMATCH")
    model_revision = identity.get("model_revision")
    tokenizer_revision = identity.get("tokenizer_revision")
    if not isinstance(model_revision, str) or not _SHA40.fullmatch(model_revision):
        raise ValueError("FROZEN_E1_MODEL_REVISION_INVALID")
    if tokenizer_revision != model_revision:
        raise ValueError("FROZEN_E1_TOKENIZER_REVISION_MISMATCH")
    if receipt.get("model_revision") != model_revision or receipt.get("model_identity_sha256") != recorded_identity_sha:
        raise ValueError("FROZEN_E1_IDENTITY_RECEIPT_MISMATCH")

    with np.load(paths["basis"], allow_pickle=False) as artifact:
        basis, mean, P, Q = (np.asarray(artifact[name]) for name in ("basis", "mean", "P", "Q"))
    with np.load(paths["fisher"], allow_pickle=False) as artifact:
        fisher = np.asarray(artifact["F_hat"])
    with np.load(paths["whitening"], allow_pickle=False) as artifact:
        whitener = np.asarray(artifact["L"])
    with np.load(paths["owner_keys"], allow_pickle=False) as artifact:
        bank = np.asarray(artifact["keys"])
    with np.load(paths["calibration_keys"], allow_pickle=False) as artifact:
        calibration_bank = np.asarray(artifact["keys"])
    key_manifest = read_json(paths["owner_key_ids"])
    calibration_manifest = read_json(paths["calibration_key_ids"])
    strength = read_json(paths["global_lambda"])
    hidden = int(identity["shapes"]["hidden_size"])
    m = int(expected_profile.scientific["m"])
    pca_dim = int(expected_profile.scientific["pca_dim"])
    if basis.shape != (hidden, pca_dim) or mean.shape != (hidden,):
        raise ValueError("FROZEN_E1_BASIS_SHAPE_MISMATCH")
    if P.shape != (hidden, m) or Q.shape != (hidden, m):
        raise ValueError("FROZEN_E1_PLANES_SHAPE_MISMATCH")
    if fisher.shape != (m, m) or whitener.shape != (m, m):
        raise ValueError("FROZEN_E1_FISHER_SHAPE_MISMATCH")
    if bank.shape != (int(expected_profile.scientific["final_keys"]), m):
        raise ValueError("FROZEN_E1_OWNER_KEYS_SHAPE_MISMATCH")
    if calibration_bank.shape != (int(expected_profile.scientific["calibration_keys"]), m):
        raise ValueError("FROZEN_E1_CALIBRATION_KEYS_SHAPE_MISMATCH")
    if not np.isfinite(basis).all() or not np.isfinite(mean).all() or not np.isfinite(P).all() or not np.isfinite(Q).all():
        raise ValueError("FROZEN_E1_GEOMETRY_NONFINITE")
    if not np.allclose(basis.T @ basis, np.eye(pca_dim), atol=2e-6, rtol=0):
        raise ValueError("FROZEN_E1_PUBLIC_BASIS_NOT_ORTHONORMAL")
    if not np.allclose(basis.T @ mean, np.zeros(pca_dim), atol=2e-6, rtol=0):
        raise ValueError("FROZEN_E1_PUBLIC_BASIS_NOT_MEAN_ORTHOGONAL")
    if not np.allclose(np.concatenate([P, Q], axis=1).T @ np.concatenate([P, Q], axis=1), np.eye(2 * m), atol=1e-6, rtol=0):
        raise ValueError("FROZEN_E1_PLANES_NOT_ORTHONORMAL")
    if not np.allclose(np.linalg.norm(bank, axis=1), 1., atol=1e-8, rtol=0):
        raise ValueError("FROZEN_E1_OWNER_KEYS_NOT_UNIT")
    actual_key_ids = key_ids(bank)
    if actual_key_ids != key_manifest.get("keys") or actual_key_ids != receipt.get("owner_key_ids"):
        raise ValueError("FROZEN_E1_OWNER_KEY_HASH_MISMATCH")
    if key_manifest.get("rule") != expected_profile.scientific["owner_key_rule"]:
        raise ValueError("FROZEN_E1_OWNER_KEY_RULE_MISMATCH")
    if key_manifest.get("owner_indices") != list(range(len(bank))):
        raise ValueError("FROZEN_E1_OWNER_KEY_ORDER_MISMATCH")
    if key_ids(calibration_bank) != calibration_manifest.get("keys"):
        raise ValueError("FROZEN_E1_CALIBRATION_KEY_HASH_MISMATCH")
    value = strength.get("lambda")
    if strength.get("status") != "FROZEN" or not isinstance(value, (float, int)) or not np.isfinite(value) or value <= 0:
        raise ValueError("FROZEN_E1_LAMBDA_INVALID")
    if float(value) != float(receipt.get("lambda")):
        raise ValueError("FROZEN_E1_LAMBDA_RECEIPT_MISMATCH")
    expected_shapes = {"hidden_size": hidden, "pca_dim": pca_dim, "m": m,
                       "owner_keys": int(expected_profile.scientific["final_keys"])}
    if receipt.get("shapes") != expected_shapes:
        raise ValueError("FROZEN_E1_RECEIPT_SHAPE_MISMATCH")
    if receipt.get("owner_key_rule") != expected_profile.scientific["owner_key_rule"]:
        raise ValueError("FROZEN_E1_OWNER_KEY_RULE_MISMATCH")
    return FrozenE1(
        profile_id=expected_profile.profile_id,
        path=root,
        model_identity=identity,
        basis=basis,
        mean=mean,
        P=P,
        Q=Q,
        fisher=fisher,
        whitener=whitener,
        lambda_value=float(value),
        key_bank=bank,
        key_ids=actual_key_ids,
        receipt=receipt,
    )
