from dataclasses import dataclass,asdict


@dataclass(frozen=True)
class E2Config:
    model: str='Qwen/Qwen2.5-3B-Instruct'
    revision: str='aa8e72537993ba99e69dfaafa59ed015b17504d1'
    tau: float=.8
    m: int=32
    pca_dim: int=128
    strength: float=.09271215438842773
    key_seed: int=2026102803
    prompt_seed: int=2026110101
    generation_seed: int=2026110102
    group_seed: int=2026110103
    owners: tuple=tuple(range(8))
    stress: tuple=(23,59)
    prompt_count: int=5000
    Q: tuple=(100,1000)
    alphas: tuple=(.01,.0001)
    max_new_tokens: int=512
    batch_size: int=8
    score_chunk: int=4
    kl_stride: int=16
    smoke_count: int=20
    quality_count: int=100
    example_count: int=10
    native_control_per_owner: int=20
    parity_max: float=.0002
    parity_p99: float=.00005
    probability_atol: float=2e-6
    negative_KL_atol: float=1e-6
    wrong_mean_concern: float=.05
    clean_Q1000_concern_count: int=2
    degeneration_increase: float=.10
    widespread_KL_key_count: int=2
    KL_exceed_fraction_concern: float=.05

    def resolved(self):
        x=asdict(self)
        for k,v in x.items():
            if isinstance(v,tuple):x[k]=list(v)
        x.update(engine='VLLM_EXACT_ROTATION_FP32',generation_backend='vllm',vllm_version='0.10.2',
            numerical_validation_policy='DISABLED_OPTIONAL_NON_BLOCKING',
            generation_linear_accumulation='float32',verification_linear_accumulation='float64',
            weights_activations_KV_dtype='float32',readout_probability_KL_dtype='float32',
            temperature_applied_once=True,top_k=0,top_p=1.,
            repetition_penalty=1.,num_beams=1,private_generated_token_control_is_secondary=True,
            severe_loop_rule='visible_tokens>=64 and repeated_4gram_fraction>=0.4',
            very_short_rule='visible_tokens<5',quality_pairing='preselected100_prompt_IDs_blinded_A_B',
            stress_group=0,wrong_key_rule='next20_cyclic_E1_indices_skipping_owner',
            KL_sample_rule='SHA256(generation_seed,prompt_id,response_position) modulo16 ==0; excludes key identity')
        return x
