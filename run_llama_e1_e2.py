"""Resume the portable Llama E1 -> E2 group0 pilot from its last checkpoint."""

from __future__ import annotations

import json
import math
import re
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path

from klrd_rs.io import digest, read_json, sha256, write_json


ROOT = Path(__file__).resolve().parent
PROFILE = "llama32_3b"
E1_RUN_ID = "run_002"
RUN_ID_PATTERN = re.compile(r"^run_(\d{3,})$")
STATE_PATH = ROOT / "runs" / PROFILE / "pipeline_resume.json"
E2_PILOT_PROMPT_COUNT = 1000
E2_BATCH_SIZE = 8
E2_SCORE_BATCH_SIZE = 32
E2_OWNERS = ("clean", *(str(index) for index in range(8)))
E2_GENERATION_BATCHES = len(E2_OWNERS) * math.ceil(E2_PILOT_PROMPT_COUNT / E2_BATCH_SIZE)
E2_SCORE_BATCHES = len(E2_OWNERS) * math.ceil(E2_PILOT_PROMPT_COUNT / E2_SCORE_BATCH_SIZE)
E2_TOTAL_RESPONSES = len(E2_OWNERS) * E2_PILOT_PROMPT_COUNT


def _e2_progress_counts(e2_root: Path) -> tuple[int, int, int, int]:
    generation_batches = 0
    score_batches = 0
    generated_responses = 0
    scored_responses = 0
    for owner in E2_OWNERS:
        generation_dir = e2_root / "pilot" / "private" / "generation" / "primary" / owner
        score_dir = e2_root / "pilot" / "private" / "scores" / "primary" / owner
        if generation_dir.is_dir():
            owner_batches = sum(1 for _ in generation_dir.glob("batch_*.json"))
            generation_batches += owner_batches
            generated_responses += min(owner_batches * E2_BATCH_SIZE, E2_PILOT_PROMPT_COUNT)
        if score_dir.is_dir():
            owner_batches = sum(
                1 for path in score_dir.glob("batch_*.npz")
                if path.with_suffix(".json").is_file()
            )
            score_batches += owner_batches
            scored_responses += min(owner_batches * E2_SCORE_BATCH_SIZE, E2_PILOT_PROMPT_COUNT)
    return (
        min(generation_batches, E2_GENERATION_BATCHES),
        min(score_batches, E2_SCORE_BATCHES),
        min(generated_responses, E2_TOTAL_RESPONSES),
        min(scored_responses, E2_TOTAL_RESPONSES),
    )


def _format_eta(seconds: float) -> str:
    seconds = max(0, round(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


class _E2Progress:
    """Aggregate generation/scoring checkpoints and estimate remaining time."""

    def __init__(self, e2_root: Path) -> None:
        self.e2_root = e2_root
        self.started = time.monotonic()
        self.previous_at = self.started
        self.generation_done, self.score_done, _, _ = _e2_progress_counts(e2_root)
        self.previous_generation = self.generation_done
        self.previous_score = self.score_done
        self.generation_rates: deque[tuple[int, float]] = deque(maxlen=20)
        self.score_rates: deque[tuple[int, float]] = deque(maxlen=20)
        self.show(force=True)

    @staticmethod
    def _rate(samples: deque[tuple[int, float]]) -> float | None:
        elapsed = sum(seconds for _, seconds in samples)
        return sum(count for count, _ in samples) / elapsed if elapsed > 0 else None

    def show(self, force: bool = False) -> None:
        now = time.monotonic()
        generation_done, score_done, generated_responses, scored_responses = _e2_progress_counts(self.e2_root)
        elapsed = now - self.previous_at
        if elapsed > 0:
            if generation_done > self.previous_generation:
                self.generation_rates.append((generation_done - self.previous_generation, elapsed))
            if score_done > self.previous_score:
                self.score_rates.append((score_done - self.previous_score, elapsed))
        self.previous_at = now
        self.previous_generation = generation_done
        self.previous_score = score_done
        self.generation_done = generation_done
        self.score_done = score_done

        generation_rate = self._rate(self.generation_rates)
        score_rate = self._rate(self.score_rates)
        generation_left = E2_GENERATION_BATCHES - generation_done
        score_left = E2_SCORE_BATCHES - score_done
        generation_eta = generation_left / generation_rate if generation_rate else None
        score_eta = score_left / score_rate if score_rate else None
        if generation_left == 0:
            generation_eta = 0.0
        if score_left == 0:
            score_eta = 0.0
        if generation_eta is not None and score_eta is not None:
            eta_text = f"overall ETA ~{_format_eta(generation_eta + score_eta)}"
        elif generation_eta is not None:
            eta_text = (f"generation ETA ~{_format_eta(generation_eta)}; "
                        "overall ETA learning scoring rate")
        elif score_eta is not None:
            eta_text = f"overall ETA ~{_format_eta(score_eta)} (scoring remaining)"
        else:
            eta_text = "ETA estimating from completed batches"

        generation_percent = 100.0 * generation_done / E2_GENERATION_BATCHES
        score_percent = 100.0 * score_done / E2_SCORE_BATCHES
        print(
            f"E2 progress: generation {generation_done:,}/{E2_GENERATION_BATCHES:,} batches "
            f"({generation_percent:.1f}%; {generated_responses:,}/{E2_TOTAL_RESPONSES:,} responses, "
            f"{E2_TOTAL_RESPONSES - generated_responses:,} left); "
            f"scoring {score_done:,}/{E2_SCORE_BATCHES:,} batches "
            f"({score_percent:.1f}%; {scored_responses:,}/{E2_TOTAL_RESPONSES:,} responses, "
            f"{E2_TOTAL_RESPONSES - scored_responses:,} left); {eta_text}",
            flush=True,
        )


def _monitor_e2_progress(e2_root: Path, stop: threading.Event) -> None:
    progress = _E2Progress(e2_root)
    while not stop.wait(30):
        progress.show()
    progress.show(force=True)


def _read_json_or_none(path: Path) -> dict | None:
    try:
        value = read_json(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _e2_source_identity() -> dict[str, str]:
    portable = ROOT / "klrd_rs" / "portable"
    return {
        "portable_e2": sha256(portable / "e2.py"),
        "vllm_adapter": sha256(portable / "vllm_adapter.py"),
        "vllm_llama": sha256(portable / "vllm_llama.py"),
        "profile_yaml": sha256(ROOT / "configs" / "model_profiles" / f"{PROFILE}.yaml"),
    }


def _existing_e2_runs() -> list[tuple[int, Path]]:
    parent = ROOT / "runs" / PROFILE / "e2"
    if not parent.is_dir():
        return []
    runs = []
    for path in parent.iterdir():
        match = RUN_ID_PATTERN.fullmatch(path.name)
        if path.is_dir() and match:
            runs.append((int(match.group(1)), path))
    return sorted(runs)


def _run_id_path(run_id: str) -> Path | None:
    if not RUN_ID_PATTERN.fullmatch(run_id):
        return None
    path = ROOT / "runs" / PROFILE / "e2" / run_id
    parent = (ROOT / "runs" / PROFILE / "e2").resolve()
    if path.resolve().parent != parent:
        return None
    return path


def _run_matches(path: Path, source_identity: dict[str, str], e1_digest: str) -> bool:
    found_identity = False
    for phase in ("smoke", "full"):
        identity_path = path / phase / "provenance" / "identity.json"
        identity = _read_json_or_none(identity_path)
        if identity is None:
            continue
        found_identity = True
        if (identity.get("source_sha256") != source_identity
                or identity.get("frozen_e1_sha256") != e1_digest):
            return False
    return found_identity


def _choose_e2_run(source_identity: dict[str, str], e1_digest: str) -> tuple[str, dict]:
    state = _read_json_or_none(STATE_PATH)
    if state:
        saved_path = _run_id_path(str(state.get("e2_run_id", "")))
        if (saved_path is not None
                and state.get("e2_source_sha256") == source_identity
                and state.get("e1_frozen_digest") == e1_digest):
            return saved_path.name, state

    for _, path in reversed(_existing_e2_runs()):
        if _run_matches(path, source_identity, e1_digest):
            return path.name, {
                "schema": "klrd_rs.portable.pipeline_resume.v1",
                "e1_run_id": E1_RUN_ID,
                "e1_frozen_digest": e1_digest,
                "e2_run_id": path.name,
                "e2_source_sha256": source_identity,
                "completed_stages": {},
            }

    existing = _existing_e2_runs()
    next_number = max((number for number, _ in existing), default=0) + 1
    run_id = f"run_{next_number:03d}"
    return run_id, {
        "schema": "klrd_rs.portable.pipeline_resume.v1",
        "e1_run_id": E1_RUN_ID,
        "e1_frozen_digest": e1_digest,
        "e2_run_id": run_id,
        "e2_source_sha256": source_identity,
        "completed_stages": {},
    }


def _e1_prepare_complete(e1_root: Path) -> bool:
    files = (
        "model_tokenizer_identity.json",
        "clean_calibration_prompts.json",
        "clean_diagnostic_prompts.json",
        "calibration_partitions.json",
        "clean_calibration.jsonl",
        "clean_diagnostic.jsonl",
    )
    if not all((e1_root / name).is_file() for name in files):
        return False
    expected = {"clean_calibration.jsonl": 2000, "clean_diagnostic.jsonl": 2500}
    for name, count in expected.items():
        with (e1_root / name).open("r", encoding="utf-8-sig") as stream:
            if sum(1 for line in stream if line.strip()) != count:
                return False
    return True


def _e1_run_complete(e1_root: Path) -> bool:
    receipt = _read_json_or_none(e1_root / "frozen_e1.json")
    summary = _read_json_or_none(e1_root / "report" / "summary.json")
    status = _read_json_or_none(e1_root / "stage_status.json")
    return bool(
        receipt and summary and status
        and receipt.get("lambda_selection", {}).get("selection_protocol_version")
        == "portable_e1_global_lambda_kl_budget_v2"
        and summary.get("status") == "FROZEN"
        and status.get("status") == "FROZEN"
    )


def _e2_phase_complete(e2_root: Path, phase: str, expected_status: str, e1_digest: str) -> bool:
    summary = _read_json_or_none(e2_root / phase / "report" / "summary.json")
    return bool(
        summary
        and summary.get("status") == expected_status
        and summary.get("frozen_e1_sha256") == e1_digest
    )


def _run_stage(name: str, arguments: list[str], complete, state: dict,
               progress_root: Path | None = None) -> None:
    if complete():
        print(f"[resume] skip completed stage: {name}", flush=True)
        state.setdefault("completed_stages", {})[name] = True
        write_json(STATE_PATH, state)
        return

    print(f"[resume] starting stage: {name}", flush=True)
    state["current_stage"] = name
    state["completed_stages"] = state.get("completed_stages", {})
    write_json(STATE_PATH, state)
    progress_stop = threading.Event()
    progress_thread = None
    if progress_root is not None:
        progress_thread = threading.Thread(
            target=_monitor_e2_progress,
            args=(progress_root, progress_stop),
            name="e2-progress",
            daemon=True,
        )
        progress_thread.start()
    try:
        result = subprocess.run([sys.executable, *arguments], cwd=ROOT, check=False)
    finally:
        if progress_thread is not None:
            progress_stop.set()
            progress_thread.join()
    if result.returncode:
        state["failed_stage"] = name
        write_json(STATE_PATH, state)
        print(f"[resume] stage failed: {name}; rerun bash run_llama_e1_e2.sh to resume", flush=True)
        raise SystemExit(result.returncode)
    if not complete():
        state["failed_stage"] = name
        write_json(STATE_PATH, state)
        raise SystemExit(f"[resume] stage returned success without completion marker: {name}")
    state.setdefault("completed_stages", {})[name] = True
    state.pop("current_stage", None)
    state.pop("failed_stage", None)
    write_json(STATE_PATH, state)


def main() -> int:
    state = _read_json_or_none(STATE_PATH) or {
        "schema": "klrd_rs.portable.pipeline_resume.v1",
        "completed_stages": {},
    }
    state["schema"] = "klrd_rs.portable.pipeline_resume.v1"
    state["e1_run_id"] = E1_RUN_ID
    e1_relative = Path("runs") / PROFILE / "e1" / E1_RUN_ID
    e1_root = ROOT / e1_relative
    _run_stage(
        "e1_prepare",
        ["main_portable_e1.py", "--profile", PROFILE, "--mode", "prepare", "--output", str(e1_relative)],
        lambda: _e1_prepare_complete(e1_root),
        state,
    )
    _run_stage(
        "e1_run",
        ["main_portable_e1.py", "--profile", PROFILE, "--mode", "run", "--output", str(e1_relative)],
        lambda: _e1_run_complete(e1_root),
        state,
    )
    _run_stage(
        "e1_report",
        ["main_portable_e1.py", "--profile", PROFILE, "--mode", "report", "--output", str(e1_relative)],
        lambda: bool(_e1_run_complete(e1_root)),
        state,
    )

    frozen_receipt = _read_json_or_none(e1_root / "frozen_e1.json")
    if frozen_receipt is None:
        raise SystemExit("[resume] E1 frozen receipt is missing; cannot begin E2")
    e1_digest = digest(frozen_receipt)
    source_identity = _e2_source_identity()
    e2_run_id, state = _choose_e2_run(source_identity, e1_digest)
    e2_relative = Path("runs") / PROFILE / "e2" / e2_run_id
    e2_root = ROOT / e2_relative
    state.update({
        "schema": "klrd_rs.portable.pipeline_resume.v1",
        "e1_run_id": E1_RUN_ID,
        "e1_frozen_digest": e1_digest,
        "e2_run_id": e2_run_id,
        "e2_source_sha256": source_identity,
    })
    write_json(STATE_PATH, state)
    print(f"[resume] E1={E1_RUN_ID}; E2={e2_run_id}", flush=True)

    e2_args = ["main_portable_e2.py", "--profile", PROFILE, "--e1", str(e1_relative)]
    _run_stage(
        "e2_smoke",
        [*e2_args, "--mode", "smoke", "--output", str(e2_relative)],
        lambda: _e2_phase_complete(e2_root, "smoke", "SMOKE_COMPLETED_DIAGNOSTIC_ONLY", e1_digest),
        state,
    )
    _run_stage(
        "e2_pilot",
        [*e2_args, "--mode", "pilot", "--output", str(e2_relative)],
        lambda: _e2_phase_complete(e2_root, "pilot", "PILOT_ONLY", e1_digest),
        state,
        progress_root=e2_root,
    )
    _run_stage(
        "e2_report",
        [*e2_args, "--mode", "report", "--output", str(e2_relative)],
        lambda: _e2_phase_complete(e2_root, "pilot", "PILOT_ONLY", e1_digest),
        state,
    )
    state.pop("current_stage", None)
    state.pop("failed_stage", None)
    write_json(STATE_PATH, state)
    print(f"[resume] pipeline pilot complete: E1={e1_relative}, E2={e2_relative}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
