from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class Config:
    model: str = 'Qwen/Qwen2.5-3B-Instruct'
    revision: str = 'aa8e72537993ba99e69dfaafa59ed015b17504d1'
    tau: float = .8
    pca_dim: int = 128
    m: int = 32
    geometry_seed: int = 2026102801
    key_calibration_seed: int = 2026102802
    key_evaluation_seed: int = 2026102803
    calibration_keys: int = 32
    validation_keys: int = 64
    fisher_context_sample_cap: int = 100000
    covariance_contexts: int = 4096
    strength_contexts: int = 128
    local_contexts: int = 512
    partition_seed: int = 2026102804
    token_sample_seed: int = 2026102805
    diagnostic_seed: int = 2026102806
    bootstrap_seed: int = 2026102807
    bootstrap_replicates: int = 2000
    diagnostic_random_keys: int = 100000
    mean_KL_target_nats: float = .005
    p99_KL_safeguard_nats: float = .05
    nominal_alpha: float = .01
    conservative_alpha: float = .0001
    query_budgets_diagnostic: tuple = (20, 100, 1000)
    ridge_fraction: float = .001
    rank_rtol: float = 1e-8
    finite_difference_h: float = .002
    finite_difference_atol: float = .003
    finite_difference_rtol: float = .03
    mean_zero_atol: float = 2e-5
    rotation_atol: float = 1e-6
    local_positive_fraction: float = .95
    lambda_max: float = 2.
    dtype: str = 'float32'
    readout_chunk: int = 4
    replay_batch_tokens: int = 4096
    replay_batch_size: int = 4
    log_interval_seconds: int = 30

    def resolved(self):
        value = asdict(self)
        value['query_budgets_diagnostic'] = list(self.query_budgets_diagnostic)
        value['calibration_prompt_partition_fractions'] = [.5, .15, .15, .2]
        value['sample_context_rule'] = 'uniform reservoir over recoverable response prediction positions, without replacement'
        value['calibration_context_eligibility'] = 'Exclude every response flagged text_roundtrip_equal=false; retain original cohort/partition IDs and record exclusions, no replacement prompts'
        value['diagnostic_tokenization_rule'] = 'Canonical text-only reencoding, including responses with nonidentical historical token roundtrip; no unknown original token IDs inferred'
        value['mean_KL_approach_relative_tolerance'] = .05
        value['Fisher_covariance_isotropy'] = 'report eigenvalues; approximate diagnostic, no post-hoc whitening'
        return value
