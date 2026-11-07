"""Portable Llama E1 preparation, geometry freeze, and read-only report."""

import hashlib
import json
import os
from types import SimpleNamespace
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from ..io import digest, read_json, read_jsonl, save_npz, sha256, write_json, write_csv
from ..math import fit_fisher, key_ids, keys, planes, sphere_p
from ..readout import RotationReadout
from .calibration import (
    build_calibration_activation_rows,
    fit_public_basis,
    partition_calibration_rows,
    prepare_clean_calibration,
)
from .frozen import load_frozen_e1
from .hf_backbone import load_backbone
from .profile import ModelProfile
from .auth import get_hf_token, resolve_hf_revision


def _seed_rows(rows: list[dict], cap: int, seed: int) -> np.ndarray:
    if not rows:
        raise ValueError("CALIBRATION_CONTEXTS_MISSING")
    offsets = np.cumsum([0] + [len(row) for row in rows])
    total = int(offsets[-1])
    if total < 2:
        raise ValueError("CALIBRATION_CONTEXTS_INSUFFICIENT")
    chosen = np.sort(np.random.default_rng(int(seed)).choice(total, min(total, int(cap)), replace=False))
    return chosen


def _selected_contexts(rows: list[dict], cap: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    counts = [len(row["residuals"]) for row in rows]
    offsets = np.cumsum([0] + counts)
    chosen = _seed_rows([row["residuals"] for row in rows], cap, seed)
    residuals, prompt_ids = [], []
    for index, row in enumerate(rows):
        pick = chosen[(chosen >= offsets[index]) & (chosen < offsets[index + 1])] - offsets[index]
        if len(pick):
            residuals.extend(row["residuals"][pick])
            prompt_ids.extend([row["prompt_id"]] * len(pick))
    return np.asarray(residuals, dtype=np.float32), np.asarray(prompt_ids, dtype=str)


def _activation_records(records: list[dict], partitions: dict[str, list[dict]], backbone: Any) -> dict[str, list[dict]]:
    from .calibration import _canonical_sha

    by_id = {row["prompt_id"]: row for row in records}
    if len(by_id) != len(records):
        raise ValueError("CLEAN_CALIBRATION_DUPLICATE_PROMPT_ID")
    expected_revision = backbone.profile.revision or getattr(backbone.model.config, "_commit_hash", None)
    if not isinstance(expected_revision, str) or len(expected_revision) != 40:
        raise ValueError("MODEL_REVISION_UNRESOLVED: activation replay requires the pinned model SHA")
    if any(row.get("model_revision") != expected_revision
           or row.get("tokenizer_revision") != expected_revision
           or row.get("generation_model_kind") != "clean" for row in records):
        raise ValueError("CLEAN_CALIBRATION_MODEL_IDENTITY_MISMATCH")
    output = {}
    for name, prompts in partitions.items():
        selected = []
        for prompt in prompts:
            record = by_id[prompt["prompt_id"]]
            if record.get("roundtrip_exact") is not True:
                continue
            encoded = build_calibration_activation_rows([record], backbone.tokenizer, backbone.max_position_embeddings)[0]
            selected.append((record, encoded))
        residual_rows = []
        chunk_size = 4
        for start in range(0, len(selected), chunk_size):
            chunk = [item[1] for item in selected[start:start + chunk_size]]
            captured = backbone.capture_pre_norm(chunk)
            for (record, encoded), residuals in zip(selected[start:start + chunk_size], captured):
                if len(residuals) != len(encoded["positions"]):
                    raise ValueError("CALIBRATION_CAPTURE_POSITION_MISMATCH")
                residual_rows.append({
                    "prompt_id": record["prompt_id"],
                    "residuals": residuals,
                    "response_sha256": record["response_sha256"],
                    "record_sha256": record["record_sha256"],
                })
        if not residual_rows:
            raise ValueError(f"CALIBRATION_PARTITION_EMPTY_AFTER_ROUNDTRIP_EXCLUSION: {name}")
        output[name] = residual_rows
    return output


def _score_visible_text_records(records: list[dict], backbone: Any, readout: RotationReadout) -> tuple[np.ndarray, list[dict]]:
    from ..data import encode_response

    vectors, metadata = [], []
    for start in range(0, len(records), 4):
        batch = records[start:start + 4]
        encoded = [encode_response(backbone.tokenizer, row["prompt"], row["response"]) for row in batch]
        for value in encoded:
            if len(value["input_ids"]) > backbone.max_position_embeddings:
                raise ValueError("DIAGNOSTIC_CONTEXT_TOO_LONG: truncation is forbidden")
        residual_rows = backbone.capture_pre_norm(encoded)
        for row, item, residuals in zip(batch, encoded, residual_rows):
            score = np.zeros(readout.p.shape[1], dtype=np.float64)
            if item["targets"]:
                for offset in range(0, len(item["targets"]), 4):
                    x = torch.as_tensor(residuals[offset:offset + 4], device=backbone.device, dtype=torch.float32)
                    target = torch.as_tensor(item["targets"][offset:offset + 4], device=backbone.device, dtype=torch.long)
                    score += readout.observed(x, target).double().cpu().numpy().sum(0)
            vectors.append(score)
            metadata.append({"prompt_id": row["prompt_id"], "visible_tokens": len(item["targets"])})
    return np.asarray(vectors, dtype=np.float64), metadata


LAMBDA_SELECTION_PROTOCOL_VERSION = "portable_e1_global_lambda_kl_budget_v2"
LAMBDA_SELECTION_RULE = "MAX_FEASIBLE_LAMBDA_UNDER_KL_BUDGETS"


def _lambda_trace_row(value: float, values: np.ndarray, target: float, p99_max: float,
                      trace_index: int) -> dict:
    raw = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = raw[np.isfinite(raw)]
    stats = np.maximum(finite, 0.0)
    mean = float(stats.mean()) if stats.size else None
    p95 = float(np.quantile(stats, 0.95)) if stats.size else None
    p99 = float(np.quantile(stats, 0.99)) if stats.size else None
    maximum = float(stats.max()) if stats.size else None
    numeric_ok = bool(
        raw.size > 0 and np.isfinite(raw).all()
        and (not finite.size or float(finite.min()) >= -1e-6)
    )
    mean_le_target = mean is not None and mean <= target
    mean_within_budget = mean is not None and mean <= target
    p99_within_limit = p99 is not None and p99 <= p99_max
    return {
        "trace_index": int(trace_index),
        "lambda": float(value),
        "measurement_count": int(raw.size),
        "mean_KL": mean,
        "p95_KL": p95,
        "p99_KL": p99,
        "max_KL": maximum,
        "p99_over_mean": (p99 / mean) if p99 is not None and mean is not None and mean > 0 else None,
        "numeric_valid": numeric_ok,
        "mean_KL_le_target": mean_le_target,
        "mean_KL_within_budget": mean_within_budget,
        "p99_KL_within_limit": p99_within_limit,
        "feasible": bool(numeric_ok and mean_within_budget and p99_within_limit),
    }


def choose_lambda_budget_constrained(
    evaluate: Callable[[float], np.ndarray],
    target: float,
    p99_max: float,
    max_lambda: float,
    iterations: int,
    trace: list[dict],
    monotonicity_tolerance: float = 1e-8,
) -> dict:
    """Choose the largest globally feasible lambda by bracket and bisection."""
    if not np.isfinite([target, p99_max, max_lambda]).all() or target <= 0 or p99_max <= 0 or max_lambda <= 0:
        raise ValueError("LAMBDA_SEARCH_CONFIGURATION_INVALID")

    def measure(value: float) -> dict:
        values = np.asarray(evaluate(float(value)), dtype=np.float64).reshape(-1)
        row = _lambda_trace_row(value, values, target, p99_max, len(trace))
        trace.append(row)
        if not row["numeric_valid"]:
            raise ValueError("LAMBDA_SEARCH_NONFINITE_OR_INVALID_KL")
        ordered = sorted(trace, key=lambda item: item["lambda"])
        for left, right in zip(ordered, ordered[1:]):
            if (right["mean_KL"] < left["mean_KL"] - monotonicity_tolerance
                    or right["p99_KL"] < left["p99_KL"] - monotonicity_tolerance):
                raise ValueError("LAMBDA_SEARCH_NON_MONOTONIC_KL_CURVE")
        return row

    zero = measure(0.0)
    if (zero["mean_KL"] is None or zero["mean_KL"] > 1e-7
            or not zero["p99_KL_within_limit"] or not zero["mean_KL_within_budget"]):
        raise ValueError("LAMBDA_ZERO_STRENGTH_PARITY_FAILED")

    low_row = zero
    high_value = min(0.01, max_lambda)
    high_row = measure(high_value)
    while high_row["feasible"] and high_value < max_lambda:
        low_row = high_row
        high_value = min(2.0 * high_value, max_lambda)
        high_row = measure(high_value)

    if high_row["feasible"] and high_value == max_lambda:
        selected = high_row
        limiting_constraint = "MAX_LAMBDA"
        max_lambda_reached = True
    else:
        max_lambda_reached = high_value == max_lambda
        low_value = float(low_row["lambda"])
        for _ in range(int(iterations)):
            midpoint = (low_value + high_value) / 2.0
            middle_row = measure(midpoint)
            if middle_row["feasible"]:
                low_value, low_row = midpoint, middle_row
            else:
                high_value, high_row = midpoint, middle_row
        selected = low_row
        violated = []
        if not high_row["mean_KL_within_budget"]:
            violated.append("MEAN_KL")
        if not high_row["p99_KL_within_limit"]:
            violated.append("P99_KL")
        limiting_constraint = "BOTH" if len(violated) == 2 else violated[0] if violated else "BOTH"

    if selected["lambda"] <= 0 or not selected["feasible"]:
        raise ValueError("LAMBDA_SEARCH_NO_POSITIVE_FEASIBLE_LAMBDA")
    return {
        "selected_lambda": float(selected["lambda"]),
        "selected_metrics": selected,
        "limiting_constraint": limiting_constraint,
        "max_lambda_reached": bool(max_lambda_reached),
        "selection_rule": LAMBDA_SELECTION_RULE,
        "selection_protocol_version": LAMBDA_SELECTION_PROTOCOL_VERSION,
    }


def _lambda_distribution(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if not values.size:
        return {"count": 0, "min": None, "mean": None, "p95": None, "p99": None, "max": None}
    return {
        "count": int(values.size),
        "min": float(values.min()),
        "mean": float(values.mean()),
        "p95": float(np.quantile(values, 0.95)),
        "p99": float(np.quantile(values, 0.99)),
        "max": float(values.max()),
    }


def _lambda_failure_reason(trace: list[dict], evaluation_errors: list[dict], error: Exception,
                           max_lambda: float, p99_max: float) -> tuple[str, list[str], dict]:
    valid = [row for row in trace if row["numeric_valid"] and row["mean_KL"] is not None]
    by_lambda = {row["lambda"]: row for row in valid}
    ordered = [by_lambda[value] for value in sorted(by_lambda)]
    mean_decreases, p99_decreases = [], []
    monotonicity_tolerance = 1e-8
    for left, right in zip(ordered, ordered[1:]):
        if right["mean_KL"] < left["mean_KL"] - monotonicity_tolerance:
            mean_decreases.append({"from_lambda": left["lambda"], "to_lambda": right["lambda"],
                                   "from_mean_KL": left["mean_KL"], "to_mean_KL": right["mean_KL"]})
        if right["p99_KL"] < left["p99_KL"] - monotonicity_tolerance:
            p99_decreases.append({"from_lambda": left["lambda"], "to_lambda": right["lambda"],
                                  "from_p99_KL": left["p99_KL"], "to_p99_KL": right["p99_KL"]})
    non_monotonic = bool(mean_decreases or p99_decreases)
    best_p99 = max((row for row in valid if row["p99_KL"] <= p99_max),
                   key=lambda row: row["lambda"], default=None)
    max_row = by_lambda.get(float(max_lambda))
    message = str(error)
    numerical_failure = bool(
        evaluation_errors or any(not row["numeric_valid"] for row in trace)
        or "NONFINITE_OR_INVALID_KL" in message
        or "POSTCHECK_FAILED" in message
    )
    if "ZERO_STRENGTH_PARITY_FAILED" in message:
        primary = "ZERO_STRENGTH_PARITY_FAILED"
    elif non_monotonic or "NON_MONOTONIC_KL_CURVE" in message:
        primary = "NON_MONOTONIC_KL_CURVE"
    elif numerical_failure:
        primary = "NUMERICAL_OR_EVALUATION_FAILURE"
    elif "CONFIGURATION_INVALID" in message:
        primary = "LAMBDA_SEARCH_CONFIGURATION_INVALID"
    else:
        primary = "NO_POSITIVE_FEASIBLE_LAMBDA"
    reasons = [primary]
    if max_row is not None and max_row["feasible"]:
        reasons.append("MAX_LAMBDA_FEASIBLE")
    return primary, reasons, {
        "mean_KL_decreases": mean_decreases,
        "p99_KL_decreases": p99_decreases,
        "monotonicity_absolute_tolerance_nats": monotonicity_tolerance,
        "p99_limit_crossed_in_trace": any(
            row["p99_KL"] is not None and row["p99_KL"] > p99_max for row in trace
        ),
        "max_lambda_was_tested": max_row is not None,
        "largest_tested_lambda_with_p99_within_limit": best_p99,
        "evaluation_errors": evaluation_errors,
    }


def _lambda_diagnostic_report(profile: ModelProfile, trace: list[dict], evaluation_errors: list[dict],
                              fisher_checks: dict, calibration_bank: np.ndarray, whitener: np.ndarray,
                              context_count: int, error: Exception | None,
                              selection: dict | None = None) -> dict:
    target = float(profile.scientific["target_mean_kl_nats"])
    p99_max = float(profile.scientific["safeguard_p99_kl_nats"])
    max_lambda, iterations, monotonicity_tolerance = 2.0, 24, 1e-8
    valid = [row for row in trace if row["numeric_valid"] and row["mean_KL"] is not None]
    best_p99 = max((row for row in valid if row["p99_KL"] <= p99_max),
                   key=lambda row: row["lambda"], default=None)
    closest_mean = min(valid, key=lambda row: (abs(row["mean_KL"] - target),
                                               not row["p99_KL_within_limit"], -row["lambda"]),
                       default=None)
    feasible = [row for row in valid if row["feasible"]]
    largest_tested = max(valid, key=lambda row: row["lambda"], default=None)

    fisher_eigenvalues = np.asarray(fisher_checks.get("eigenvalues", []), dtype=np.float64)
    fisher_effective = np.asarray(fisher_checks.get("effective_eigenvalues", []), dtype=np.float64)
    fisher = {
        "rank": int(fisher_checks.get("rank", 0)),
        "min_eigenvalue": float(fisher_eigenvalues.min()) if fisher_eigenvalues.size else None,
        "max_eigenvalue": float(fisher_eigenvalues.max()) if fisher_eigenvalues.size else None,
        "condition_number": fisher_checks.get("condition_number"),
        "ridge": fisher_checks.get("ridge"),
        "effective_min_eigenvalue": float(fisher_effective.min()) if fisher_effective.size else None,
        "effective_max_eigenvalue": float(fisher_effective.max()) if fisher_effective.size else None,
        "effective_condition_number": fisher_checks.get("effective_condition_number"),
    }
    directions = np.asarray([whitener @ key for key in calibration_bank], dtype=np.float64)
    important_lambdas = {
        "largest_tested_p99_feasible": best_p99["lambda"] if best_p99 else None,
        "closest_to_mean_target": closest_mean["lambda"] if closest_mean else None,
        "largest_lambda_tested": largest_tested["lambda"] if largest_tested else None,
        "max_lambda_search_limit": max_lambda,
        "selected_lambda": selection["selected_lambda"] if selection else None,
    }
    rotations = {}
    for label, value in important_lambdas.items():
        if value is None:
            rotations[label] = None
            continue
        angles = float(value) * directions
        rotations[label] = {
            "lambda": float(value),
            "Lk_norm": _lambda_distribution(np.linalg.norm(directions, axis=1)),
            "lambda_Lk_norm": _lambda_distribution(np.linalg.norm(angles, axis=1)),
            "max_absolute_angle": float(np.abs(angles).max()) if angles.size else None,
        }

    failure = None
    if error is not None:
        primary, reasons, curve = _lambda_failure_reason(
            trace, evaluation_errors, error, max_lambda, p99_max,
        )
        failure = {
            "exception_type": type(error).__name__,
            "message": str(error),
            "primary_cause": primary,
            "causes": reasons,
            "curve_diagnostics": curve,
        }
    measurements_per_lambda = int(len(calibration_bank) * context_count)
    selected_metrics = selection["selected_metrics"] if selection else {}
    report = {
        "schema": "klrd_rs.portable.lambda_search_diagnostic.v2",
        "status": "FAILED" if error is not None else "FEASIBLE",
        "failure": failure,
        "selection_protocol_version": LAMBDA_SELECTION_PROTOCOL_VERSION,
        "selection_rule": LAMBDA_SELECTION_RULE,
        "selected_lambda": selection["selected_lambda"] if selection else None,
        "mean_KL": selected_metrics.get("mean_KL"),
        "p95_KL": selected_metrics.get("p95_KL"),
        "p99_KL": selected_metrics.get("p99_KL"),
        "max_KL": selected_metrics.get("max_KL"),
        "mean_KL_budget": target,
        "p99_KL_budget": p99_max,
        "limiting_constraint": selection["limiting_constraint"] if selection else None,
        "max_lambda_reached": selection["max_lambda_reached"] if selection else False,
        "configuration": {
            "target_mean_KL": target,
            "p99_max": p99_max,
            "max_lambda": max_lambda,
            "iterations": iterations,
            "monotonicity_absolute_tolerance_nats": monotonicity_tolerance,
            "fisher_ridge_fraction": float(profile.scientific["fisher_ridge_fraction"]),
        },
        "summary": {
            "largest_tested_lambda_with_p99_within_limit": best_p99,
            "lambda_closest_to_mean_target": closest_mean,
            "feasible_lambda_found": max(feasible, key=lambda row: row["lambda"], default=None),
        },
        "fisher_diagnostics": fisher,
        "rotation_diagnostics": rotations,
        "calibration_statistics": {
            "calibration_key_count": int(len(calibration_bank)),
            "strength_context_count": int(context_count),
            "KL_measurements_per_lambda": measurements_per_lambda,
            "lambda_evaluations_completed": int(len(trace)),
            "total_KL_measurements": int(sum(row["measurement_count"] for row in trace)),
            "evaluation_errors": evaluation_errors,
        },
        "lambda_search_trace": trace,
    }
    return report


def freeze_e1_geometry(
    profile: ModelProfile,
    output: str | Path,
    basis: np.ndarray,
    mean: np.ndarray,
    fisher_scores: np.ndarray,
    lambda_evaluator: Callable[[float, np.ndarray, np.ndarray], np.ndarray],
    model_identity: dict,
    run_inputs: dict | None = None,
    calibration_context_count: int | None = None,
) -> dict:
    """Freeze public geometry and one global lambda, then draw the owner bank."""
    profile.validate()
    root = Path(output)
    receipt_path = root / "frozen_e1.json"
    if receipt_path.exists():
        raise ValueError("FROZEN_E1_IMMUTABLE: existing receipt cannot be overwritten; choose a new run directory")
    m = int(profile.scientific["m"])
    pca_dim = int(profile.scientific["pca_dim"])
    basis, mean = np.asarray(basis, dtype=np.float64), np.asarray(mean, dtype=np.float64)
    if basis.ndim != 2 or basis.shape[1] != pca_dim or mean.shape != (basis.shape[0],):
        raise ValueError("E1_PUBLIC_BASIS_SHAPE_MISMATCH")
    if model_identity.get("profile_id") != profile.profile_id or model_identity.get("model_id") != profile.model_id:
        raise ValueError("E1_MODEL_IDENTITY_PROFILE_MISMATCH")
    model_revision = model_identity.get("model_revision")
    if not isinstance(model_revision, str) or len(model_revision) != 40:
        raise ValueError("E1_MODEL_REVISION_UNRESOLVED")

    root.mkdir(parents=True, exist_ok=True)
    geometry_dir = root / "geometry"
    private_dir = root / "private"
    keys_dir = root / "keys"
    teacher_dir = root / "teacher"
    for directory in (geometry_dir, private_dir, keys_dir, teacher_dir):
        directory.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(private_dir, 0o700)
    except OSError:
        pass

    P, Q = planes(basis, m, int(profile.seeds["geometry"]))
    frame = np.concatenate([P, Q], axis=1)
    if not np.allclose(frame.T @ frame, np.eye(2 * m), atol=1e-6, rtol=0):
        raise ValueError("E1_ROTATION_PLANES_NOT_ORTHONORMAL")
    save_npz(geometry_dir / "basis.npz", basis=basis, mean=mean, P=P, Q=Q)
    fisher, whitener, fisher_checks = fit_fisher(np.asarray(fisher_scores, dtype=np.float64))
    save_npz(geometry_dir / "fisher.npz", F_hat=fisher, F_eff=fisher + fisher_checks["ridge"] * np.eye(m))
    save_npz(geometry_dir / "whitening.npz", L=whitener)
    write_json(geometry_dir / "fisher_checks.json", fisher_checks)

    calibration_bank = keys(int(profile.scientific["calibration_keys"]), m, int(profile.seeds["calibration_keys"]))
    save_npz(private_dir / "calibration_keys.npz", keys=calibration_bank)
    write_json(keys_dir / "calibration_key_ids.json", {
        "seed": int(profile.seeds["calibration_keys"]), "keys": key_ids(calibration_bank),
        "vectors_path": "private/calibration_keys.npz",
    })
    target_mean = float(profile.scientific["target_mean_kl_nats"])
    p99_max = float(profile.scientific["safeguard_p99_kl_nats"])
    max_lambda, iterations = 2.0, 24
    lambda_trace: list[dict] = []
    lambda_evaluation_errors: list[dict] = []

    def tracked_lambda_evaluator(value: float, key_bank: np.ndarray, L: np.ndarray) -> np.ndarray:
        try:
            measurements = np.asarray(lambda_evaluator(value, key_bank, L), dtype=np.float64)
        except Exception as exc:
            lambda_evaluation_errors.append({
                "lambda": float(value), "exception_type": type(exc).__name__, "message": str(exc),
            })
            raise
        return measurements

    try:
        selection = choose_lambda_budget_constrained(
            lambda value: tracked_lambda_evaluator(value, calibration_bank, whitener),
            target=target_mean, p99_max=p99_max, max_lambda=max_lambda,
            iterations=iterations, trace=lambda_trace,
        )
        strength = selection["selected_lambda"]
        selected_metrics = selection["selected_metrics"]
        if not (selected_metrics["mean_KL"] <= target_mean
                and selected_metrics["p99_KL"] <= p99_max and selected_metrics["feasible"]):
            raise ValueError("LAMBDA_SELECTION_BUDGET_POSTCHECK_FAILED")
    except Exception as exc:
        measured_context_count = calibration_context_count
        if measured_context_count is None and lambda_trace and len(calibration_bank):
            measured_context_count = lambda_trace[0]["measurement_count"] // len(calibration_bank)
        diagnostic = _lambda_diagnostic_report(
            profile, lambda_trace, lambda_evaluation_errors, fisher_checks,
            calibration_bank, whitener, int(measured_context_count or 0), exc,
        )
        write_json(teacher_dir / "lambda_search_diagnostic.json", diagnostic)
        best_p99 = max(
            (row for row in lambda_trace if row["feasible"]),
            key=lambda row: row["lambda"], default=None,
        )
        closest = diagnostic["summary"]["lambda_closest_to_mean_target"]
        print(
            "[E1 lambda search] FAILED: "
            f"{diagnostic['failure']['primary_cause']}; "
            f"largest tested budget-feasible lambda={best_p99['lambda'] if best_p99 else None}, "
            f"mean_KL={best_p99['mean_KL'] if best_p99 else None}, "
            f"p99_KL={best_p99['p99_KL'] if best_p99 else None}; "
            f"closest-to-target lambda={closest['lambda'] if closest else None}, "
            f"mean_KL={closest['mean_KL'] if closest else None}, "
            f"p99_KL={closest['p99_KL'] if closest else None}. "
            f"Full diagnostics: {teacher_dir / 'lambda_search_diagnostic.json'}"
        )
        raise

    measured_context_count = calibration_context_count
    if measured_context_count is None and lambda_trace and len(calibration_bank):
        measured_context_count = lambda_trace[0]["measurement_count"] // len(calibration_bank)
    diagnostic = _lambda_diagnostic_report(
        profile, lambda_trace, lambda_evaluation_errors, fisher_checks,
        calibration_bank, whitener, int(measured_context_count or 0), None,
        selection=selection,
    )
    legacy_trace = [
        {"lambda": row["lambda"], "mean": row["mean_KL"], "p99": row["p99_KL"]}
        for row in lambda_trace
    ]
    diagnostic["legacy_choose_lambda_trace"] = legacy_trace
    write_json(teacher_dir / "lambda_search_diagnostic.json", diagnostic)
    strength_record = {
        "schema": "klrd_rs.portable.global_lambda.v1",
        "status": "FROZEN",
        "lambda": float(strength),
        "pooled_search_trace": legacy_trace,
        "calibration_keys": int(profile.scientific["calibration_keys"]),
        "global_only": True,
        "owner_keys_sampled_after_freeze": True,
        "KL_direction": "KL(q||p0)",
        "mean_target": float(profile.scientific["target_mean_kl_nats"]),
        "p99_max": float(profile.scientific["safeguard_p99_kl_nats"]),
        "selected_lambda": float(strength),
        "mean_KL": selected_metrics["mean_KL"],
        "p95_KL": selected_metrics["p95_KL"],
        "p99_KL": selected_metrics["p99_KL"],
        "max_KL": selected_metrics["max_KL"],
        "mean_KL_budget": target_mean,
        "p99_KL_budget": p99_max,
        "limiting_constraint": selection["limiting_constraint"],
        "selection_rule": LAMBDA_SELECTION_RULE,
        "selection_protocol_version": LAMBDA_SELECTION_PROTOCOL_VERSION,
    }
    write_json(teacher_dir / "global_lambda.json", strength_record)

    # The final owner stream is intentionally sampled only after the lambda file is durable.
    owner_bank = keys(int(profile.scientific["final_keys"]), m, int(profile.seeds["final_keys"]))
    save_npz(private_dir / "owner_keys.npz", keys=owner_bank)
    owner_ids = key_ids(owner_bank)
    write_json(keys_dir / "owner_key_ids.json", {
        "seed": int(profile.seeds["final_keys"]),
        "rule": profile.scientific["owner_key_rule"],
        "owner_indices": list(range(len(owner_bank))),
        "keys": owner_ids,
        "vectors_path": "private/owner_keys.npz",
    })

    identity_path = root / "model_tokenizer_identity.json"
    if not identity_path.is_file():
        write_json(identity_path, model_identity)
    elif read_json(identity_path) != model_identity:
        raise ValueError("E1_MODEL_IDENTITY_RECEIPT_MISMATCH")
    artifacts = {
        "model_identity": "model_tokenizer_identity.json",
        "basis": "geometry/basis.npz",
        "fisher": "geometry/fisher.npz",
        "whitening": "geometry/whitening.npz",
        "global_lambda": "teacher/global_lambda.json",
        "lambda_search_diagnostic": "teacher/lambda_search_diagnostic.json",
        "calibration_keys": "private/calibration_keys.npz",
        "calibration_key_ids": "keys/calibration_key_ids.json",
        "owner_keys": "private/owner_keys.npz",
        "owner_key_ids": "keys/owner_key_ids.json",
    }
    artifact_manifest = {name: {"path": relative, "sha256": sha256(root / relative)} for name, relative in artifacts.items()}
    receipt = {
        "schema": "klrd_rs.portable.frozen_e1.v1",
        "profile_id": profile.profile_id,
        "model_id": profile.model_id,
        "model_revision": model_revision,
        "tokenizer_revision": model_identity.get("tokenizer_revision"),
        "model_identity_sha256": model_identity.get("identity_sha256"),
        "lambda": float(strength),
        "lambda_selection": {
            "selected_lambda": float(strength),
            "mean_KL": selected_metrics["mean_KL"],
            "p95_KL": selected_metrics["p95_KL"],
            "p99_KL": selected_metrics["p99_KL"],
            "max_KL": selected_metrics["max_KL"],
            "mean_KL_budget": target_mean,
            "p99_KL_budget": p99_max,
            "limiting_constraint": selection["limiting_constraint"],
            "selection_rule": LAMBDA_SELECTION_RULE,
            "selection_protocol_version": LAMBDA_SELECTION_PROTOCOL_VERSION,
        },
        "shapes": {"hidden_size": int(basis.shape[0]), "pca_dim": int(basis.shape[1]), "m": m,
                    "owner_keys": int(len(owner_bank))},
        "owner_key_rule": profile.scientific["owner_key_rule"],
        "owner_key_ids": owner_ids,
        "geometry_strength_sha256": digest({"P": P.tolist(), "Q": Q.tolist(), "L": whitener.tolist(), "lambda": strength}),
        "run_inputs": run_inputs or {},
        "artifacts": artifact_manifest,
    }
    write_json(receipt_path, receipt)
    return {"status": "FROZEN", "lambda": float(strength), "key_ids": owner_ids, "receipt": str(receipt_path), "fisher_checks": fisher_checks}


def run_e1(profile: ModelProfile, output: str | Path, mode: str) -> dict:
    """Run explicit E1 prepare/run phases or return a read-only report."""
    profile.validate()
    repo_root = Path(__file__).resolve().parents[2]
    root = profile.validate_run_path(repo_root, "e1", Path(output))
    if mode == "prepare":
        from transformers import AutoConfig, AutoTokenizer

        token = get_hf_token(repo_root)
        revision = profile.revision
        if revision is None:
            revision = resolve_hf_revision(profile.model_id, token)
        config = AutoConfig.from_pretrained(
            profile.model_id, revision=revision, trust_remote_code=False, token=token,
        )
        tokenizer = AutoTokenizer.from_pretrained(profile.model_id, revision=revision, trust_remote_code=False, token=token)
        config_revision = getattr(config, "_commit_hash", None)
        if config_revision is not None and config_revision != revision:
            raise ValueError("MODEL_REVISION_DRIFT: AutoConfig did not resolve the requested pinned commit")
        from dataclasses import replace
        pinned_profile = profile if profile.revision == revision else replace(profile, revision=revision)
        metadata_backbone = SimpleNamespace(
            profile=SimpleNamespace(revision=revision),
            tokenizer=tokenizer,
            model=SimpleNamespace(config=config, generation_config=AutoConfig),
        )
        # Load the model generation config separately from the immutable checkpoint revision.
        from transformers import GenerationConfig
        metadata_backbone.model.generation_config = GenerationConfig.from_pretrained(
            profile.model_id, revision=revision, token=token,
        )
        calibration = prepare_clean_calibration(pinned_profile, metadata_backbone, root, "calibration")
        diagnostic = prepare_clean_calibration(pinned_profile, metadata_backbone, root, "diagnostic")
        partitions = partition_calibration_rows(
            read_json(root / "clean_calibration_prompts.json")["prompts"],
            seed=int(profile.seeds["calibration_partition"]),
        )
        write_json(root / "calibration_partitions.json", {
            "seed": int(profile.seeds["calibration_partition"]),
            "roles": {name: [row["prompt_id"] for row in rows] for name, rows in partitions.items()},
            "source": "fresh clean Llama calibration prompts only",
        })
        return {"status": "PREPARED", "calibration": calibration, "diagnostic": diagnostic}
    if mode == "report":
        receipt = load_frozen_e1(root, profile)
        report = {
            "status": "FROZEN",
            "profile_id": receipt.profile_id,
            "model_revision": receipt.model_identity["model_revision"],
            "lambda": receipt.lambda_value,
            "geometry_shapes": receipt.receipt["shapes"],
            "owner_key_count": len(receipt.key_ids),
            "server_parity": "NOT_RUN",
            "scientific_scope": "E1 calibration freeze only; no E2 detection or fixed-key FPR claim",
        }
        write_json(root / "report" / "summary.json", report)
        return report
    if mode != "run":
        raise ValueError("E1 mode must be prepare, run, or report")
    if (root / "frozen_e1.json").exists():
        existing_frozen = load_frozen_e1(root, profile)
    else:
        existing_frozen = None

    identity = read_json(root / "model_tokenizer_identity.json")
    if identity.get("profile_id") != profile.profile_id or identity.get("model_id") != profile.model_id:
        raise ValueError("E1_MODEL_IDENTITY_MISMATCH")
    if profile.revision is not None and identity.get("model_revision") != profile.revision:
        raise ValueError("E1_MODEL_REVISION_MISMATCH")
    calibration_path = root / "clean_calibration.jsonl"
    if not calibration_path.is_file():
        raise ValueError(f"MISSING_CLEAN_CALIBRATION: run --mode prepare first ({calibration_path})")
    records = read_jsonl(calibration_path)
    expected_count = int(profile.scientific["calibration_responses"])
    if len(records) != expected_count:
        raise ValueError(f"CLEAN_CALIBRATION_COUNT_MISMATCH: expected {expected_count}, got {len(records)}")
    prompt_manifest = read_json(root / "clean_calibration_prompts.json")["prompts"]
    expected_ids = [row["prompt_id"] for row in prompt_manifest]
    if len(prompt_manifest) != expected_count or [row.get("prompt_id") for row in records] != expected_ids:
        raise ValueError("CLEAN_CALIBRATION_PROMPT_MANIFEST_MISMATCH")
    if any(
        row.get("response_sha256") != hashlib.sha256(str(row.get("response", "")).encode()).hexdigest()
        or row.get("record_sha256") != digest({key: value for key, value in row.items() if key != "record_sha256"})
        or row.get("model_revision") != identity.get("model_revision")
        or row.get("tokenizer_revision") != identity.get("tokenizer_revision")
        or row.get("generation_model_kind") != "clean"
        for row in records
    ):
        raise ValueError("CLEAN_CALIBRATION_IDENTITY_OR_CHECKSUM_MISMATCH")
    diagnostic_path = root / "clean_diagnostic.jsonl"
    if not diagnostic_path.is_file():
        raise ValueError(f"MISSING_CLEAN_DIAGNOSTIC: run --mode prepare first ({diagnostic_path})")
    diagnostic_records = read_jsonl(diagnostic_path)
    expected_diagnostic_count = int(profile.scientific["clean_original_diagnostic_responses"])
    if len(diagnostic_records) != expected_diagnostic_count:
        raise ValueError(f"CLEAN_DIAGNOSTIC_COUNT_MISMATCH: expected {expected_diagnostic_count}, got {len(diagnostic_records)}")
    if any(
        row.get("response_sha256") != hashlib.sha256(str(row.get("response", "")).encode()).hexdigest()
        or row.get("record_sha256") != digest({key: value for key, value in row.items() if key != "record_sha256"})
        or row.get("model_revision") != identity.get("model_revision")
        or row.get("tokenizer_revision") != identity.get("tokenizer_revision")
        for row in diagnostic_records
    ):
        raise ValueError("CLEAN_DIAGNOSTIC_IDENTITY_OR_CHECKSUM_MISMATCH")
    from dataclasses import replace
    pinned_profile = profile if profile.revision == identity.get("model_revision") else replace(profile, revision=identity.get("model_revision"))
    backbone = load_backbone(pinned_profile, device="cuda")
    loaded_revision = getattr(backbone.model.config, "_commit_hash", None)
    if loaded_revision is not None and loaded_revision != identity.get("model_revision"):
        raise ValueError("E1_LOADED_MODEL_REVISION_MISMATCH")
    partitions = partition_calibration_rows(prompt_manifest, int(profile.seeds["calibration_partition"]))
    partitions_path = root / "calibration_partitions.json"
    expected_partitions = {
        "seed": int(profile.seeds["calibration_partition"]),
        "roles": {name: [row["prompt_id"] for row in values] for name, values in partitions.items()},
        "source": "fresh clean Llama calibration prompts only",
    }
    if partitions_path.exists() and read_json(partitions_path) != expected_partitions:
        raise ValueError("E1_PARTITION_IDENTITY_DRIFT")
    write_json(partitions_path, expected_partitions)
    source_files = [
        Path(__file__), Path(__file__).with_name("calibration.py"), Path(__file__).with_name("hf_backbone.py"),
        Path(__file__).with_name("profile.py"), Path(__file__).parents[1] / "math.py",
        Path(__file__).parents[1] / "readout.py", Path(__file__).parents[1] / "data.py",
        Path(__file__).parents[1] / "io.py",
    ]
    run_inputs = {
        "model_identity_sha256": identity.get("identity_sha256"),
        "calibration_responses_sha256": sha256(calibration_path),
        "clean_diagnostic_sha256": sha256(root / "clean_diagnostic.jsonl"),
        "diagnostic_prompt_manifest_sha256": sha256(root / "clean_diagnostic_prompts.json"),
        "prompt_manifest_sha256": sha256(root / "clean_calibration_prompts.json"),
        "partition_manifest_sha256": sha256(partitions_path),
        "profile_sha256": sha256(Path(__file__).parents[2] / "configs" / "model_profiles" / f"{profile.profile_id}.yaml"),
        "source_sha256": {str(path.name): sha256(path) for path in source_files},
    }
    if existing_frozen is not None and existing_frozen.receipt.get("run_inputs") != run_inputs:
        raise ValueError("FROZEN_E1_RESUME_IDENTITY_DRIFT: frozen geometry inputs or source changed")
    activations = _activation_records(records, partitions, backbone)
    caps = profile.scientific["calibration_context_caps"]
    context_seed = int(profile.seeds["context_sampling"])
    pca_contexts, _pca_ids = _selected_contexts(
        activations["pca_covariance"], cap=int(caps["pca_covariance"]), seed=context_seed,
    )
    mean = pca_contexts.mean(axis=0, dtype=np.float64)
    basis, mean = fit_public_basis(pca_contexts, mean, int(profile.scientific["pca_dim"]), int(profile.seeds["geometry"]))
    readout = backbone.readout(*planes(basis, int(profile.scientific["m"]), int(profile.seeds["geometry"])),
                               float(profile.scientific["tau"]))
    native_tail_error = backbone.native_tail_equivalence_check(pca_contexts[:2])
    if not np.isfinite(native_tail_error) or native_tail_error > 2e-4:
        raise ValueError(f"NATIVE_HF_TAIL_PARITY_FAILED: max_abs_error={native_tail_error}")
    write_json(root / "geometry" / "native_tail_checks.json", {
        "status": "PASS", "max_abs_error": native_tail_error,
        "residuals": min(2, len(pca_contexts)), "reference": "native final RMSNorm then native lm_head",
    })

    fisher_contexts, _ = _selected_contexts(activations["fisher"], cap=int(caps["fisher"]), seed=context_seed + 1)
    fisher_readout = backbone.readout(readout.p, readout.q, float(profile.scientific["tau"]))
    fisher_scores = []
    generator = torch.Generator(device=backbone.device).manual_seed(int(profile.seeds["token_sampling"]))
    for start in range(0, len(fisher_contexts), 4):
        residuals = torch.as_tensor(fisher_contexts[start:start + 4], device=backbone.device, dtype=torch.float32)
        fisher_scores.extend(fisher_readout.observed(residuals, generator=generator).double().cpu().numpy())

    strength_contexts, _ = _selected_contexts(activations["strength"], cap=int(caps["strength"]), seed=context_seed + 2)
    strength_tensor = torch.as_tensor(strength_contexts, device=backbone.device, dtype=torch.float32)
    def evaluate_strength(value, calibration_bank, whitener):
        values = []
        for key in calibration_bank:
            angle = torch.as_tensor(float(value) * (whitener @ key), device=backbone.device, dtype=torch.float32)
            for start in range(0, len(strength_tensor), 4):
                values.extend(readout.kl(strength_tensor[start:start + 4], angle).double().cpu().numpy().tolist())
        return np.asarray(values, dtype=np.float64)

    if existing_frozen is None:
        result = freeze_e1_geometry(
            profile, root, basis, mean, np.asarray(fisher_scores), evaluate_strength, identity,
            run_inputs=run_inputs, calibration_context_count=len(strength_contexts),
        )
    else:
        result = {
            "status": "FROZEN",
            "lambda": existing_frozen.lambda_value,
            "key_ids": existing_frozen.key_ids,
            "receipt": str(root / "frozen_e1.json"),
            "fisher_checks": read_json(root / "geometry" / "fisher_checks.json"),
        }
        basis, mean = existing_frozen.basis, existing_frozen.mean
        readout = backbone.readout(existing_frozen.P, existing_frozen.Q, float(profile.scientific["tau"]))
    owner_keys_path = root / "private" / "owner_keys.npz"
    with np.load(owner_keys_path, allow_pickle=False) as artifact:
        owner_bank = artifact["keys"]
    with np.load(root / "geometry" / "whitening.npz", allow_pickle=False) as artifact:
        whitener = artifact["L"]
    covariance_contexts, _ = _selected_contexts(
        activations["pca_covariance"], cap=int(caps["pca_covariance"]), seed=context_seed + 3,
    )
    covariance_tensor = torch.as_tensor(covariance_contexts, device=backbone.device, dtype=torch.float32)
    covariance_generator = torch.Generator(device=backbone.device).manual_seed(int(profile.seeds["token_sampling"]) + 1)
    covariance_scores = []
    for start in range(0, len(covariance_tensor), 4):
        scores = readout.observed(covariance_tensor[start:start + 4], generator=covariance_generator)
        covariance_scores.extend(scores.double().cpu().numpy())
    whitened_covariance = np.asarray(covariance_scores) @ whitener.T
    covariance_matrix = whitened_covariance.T @ whitened_covariance / len(whitened_covariance)
    write_json(root / "geometry" / "heldout_covariance_checks.json", {
        "observations": len(covariance_scores),
        "whitened_eigenvalues": np.linalg.eigvalsh(covariance_matrix).tolist(),
        "whitened_covariance_frobenius_error": float(np.linalg.norm(covariance_matrix - np.eye(int(profile.scientific["m"])))),
        "interpretation": "independent clean-token covariance diagnostic; no post-hoc whitening",
    })
    local_contexts, local_prompt_ids = _selected_contexts(
        activations["local"], cap=int(caps["local"]), seed=context_seed + 4,
    )
    local_tensor = torch.as_tensor(local_contexts, device=backbone.device, dtype=torch.float32)
    local_summary = []
    for index, key in enumerate(owner_bank):
        direction = torch.as_tensor(whitener @ key, device=backbone.device, dtype=torch.float32)
        delta, kl, prediction, finite_difference, noise = [], [], [], [], []
        for start in range(0, len(local_tensor), 4):
            values = readout.local(local_tensor[start:start + 4], direction, result["lambda"])
            for target, part in zip((delta, kl, prediction, finite_difference, noise), values):
                target.extend(part.double().cpu().numpy().tolist())
        local_summary.append({
            "key_index": index, "Delta": float(np.mean(delta)), "mean_KL": float(np.mean(kl)),
            "p99_KL": float(np.quantile(kl, .99)), "first_order_prediction": float(np.mean(prediction)),
            "finite_difference_slope": float(np.mean(finite_difference)),
            "per_token_clean_noise_RMS": float(np.sqrt(np.mean(np.square(noise)))),
            "independent_prompts": int(len(set(local_prompt_ids))), "contexts": int(len(delta)),
            "mean_KL_violation": float(np.mean(kl)) > float(profile.scientific["target_mean_kl_nats"]),
            "p99_KL_violation": float(np.quantile(kl, .99)) > float(profile.scientific["safeguard_p99_kl_nats"]),
        })
    write_json(root / "teacher" / "local_capacity_by_key.json", {
        "keys_filtered": False, "rows": local_summary,
        "status": "DESCRIPTIVE_LOCAL_TEACHER_ONLY",
    })
    diagnostic_phi, diagnostic_metadata = _score_visible_text_records(diagnostic_records, backbone, readout)
    diagnostic_psi = diagnostic_phi @ whitener.T
    save_npz(root / "private" / "clean_original_score_vectors.npz", vectors=diagnostic_psi,
             prompt_ids=np.asarray([row["prompt_id"] for row in diagnostic_records], dtype=str),
             visible_token_counts=np.asarray([row["visible_tokens"] for row in diagnostic_metadata], dtype=np.int64))
    diagnostic_rows, diagnostic_summary = [], {}
    for budget in (20, 100, 1000):
        groups_count = len(diagnostic_psi) // budget
        key_probabilities = []
        for group_index in range(groups_count):
            vector = diagnostic_psi[group_index * budget:(group_index + 1) * budget].sum(0)
            probabilities = np.asarray(sphere_p(vector, owner_bank), dtype=np.float64)
            key_probabilities.append(probabilities)
            norm = float(np.linalg.norm(vector))
            cosines = owner_bank @ vector / norm if norm else np.zeros(len(owner_bank))
            diagnostic_rows.extend({
                "query_budget": budget,
                "group_index": group_index,
                "key_index": key_index,
                "response_count": budget,
                "vector_norm": norm,
                "cosine": float(cosines[key_index]),
                "p_value": float(probabilities[key_index]),
            } for key_index in range(len(owner_bank)))
        pvalue_matrix = np.asarray(key_probabilities, dtype=np.float64)
        diagnostic_summary[str(budget)] = {
            "independent_disjoint_groups": groups_count,
            "discarded_responses": len(diagnostic_psi) - groups_count * budget,
            "per_key_pvalue_counts_at_alpha": {
                str(alpha): [int((pvalue_matrix[:, key_index] <= alpha).sum()) for key_index in range(len(owner_bank))]
                for alpha in profile.scientific["alphas"]
            },
            "scope": "descriptive clean-original diagnostic only; insufficient to certify fixed-key FPR",
        }
    write_csv(root / "null" / "clean_original_key_pvalues.csv", diagnostic_rows,
              ["query_budget", "group_index", "key_index", "response_count", "vector_norm", "cosine", "p_value"])
    write_json(root / "null" / "clean_original_summary.json", {
        "status": "DESCRIPTIVE_ONLY", "responses": len(diagnostic_psi),
        "groups": diagnostic_summary, "clean_paraphrase": "NOT_AVAILABLE",
        "key_independent_fixed_vector_null": "analytic spherical p-value; no clean threshold fitting",
    })
    stage_status = {
        "status": "FROZEN",
        "stages": {
            "clean_calibration": {"status": "PASS", "response_count": len(records), "source": profile.data["calibration_split"]},
            "public_basis": {"status": "PASS", "pca_dim": int(profile.scientific["pca_dim"])},
            "fisher": {"status": "PASS", "observations": len(fisher_scores)},
            "global_lambda": {"status": "FROZEN", "value": result["lambda"]},
            "owner_keys": {"status": "FROZEN_AFTER_LAMBDA", "count": len(owner_bank)},
            "local_teacher": {"status": "DESCRIPTIVE_ONLY", "rows": len(local_summary)},
            "clean_original_null": {"status": "DESCRIPTIVE_ONLY", "responses": len(diagnostic_psi)},
            "clean_paraphrase": {"status": "NOT_AVAILABLE"},
            "server_vllm_parity": {"status": "NOT_RUN"},
        },
    }
    write_json(root / "stage_status.json", stage_status)
    report = {
        "status": "FROZEN",
        "profile_id": profile.profile_id,
        "model_revision": identity["model_revision"],
        "lambda": result["lambda"],
        "owner_key_count": len(owner_bank),
        "local_positive_key_fraction": float(np.mean([row["Delta"] > 0 for row in local_summary])),
        "local_KL_violation_key_indices": [row["key_index"] for row in local_summary
                                           if row["mean_KL_violation"] or row["p99_KL_violation"]],
        "clean_original_diagnostic_responses": len(diagnostic_psi),
        "clean_paraphrase": "NOT_AVAILABLE",
        "server_parity": "NOT_RUN",
        "scientific_scope": "E1 local teacher feasibility only; no E2 detection or fixed-key FPR certification",
    }
    write_json(root / "report" / "summary.json", report)
    return report
