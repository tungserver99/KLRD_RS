"""Frozen scientific identity and predeclared single-key LoRA schedule."""
from dataclasses import dataclass,asdict


@dataclass(frozen=True)
class E3Config:
    profile: str='pilot'
    model: str='Qwen/Qwen2.5-3B-Instruct'
    revision: str='aa8e72537993ba99e69dfaafa59ed015b17504d1'
    dataset: str='HuggingFaceH4/ultrachat_200k'
    dataset_revision: str='8049631c405ae6576f93f445c6b8166f76f5505a'
    tau: float=.8
    m: int=32
    pca_dim: int=128
    strength: float=.09271215438842773
    key_seed: int=2026102803
    owner_key_id: int=0
    alphas: tuple=(.01,.0001)
    seed: int=2026110301
    max_new_tokens: int=512
    lora_r: int=16
    lora_alpha: int=32
    lora_dropout: float=0.
    target_modules: tuple=('q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj')
    learning_rate: float=1e-5
    max_epochs: int=2
    per_device_train_batch_size: int=1
    gradient_accumulation_steps: int=16
    warmup_ratio: float=.03
    weight_decay: float=0.
    max_grad_norm: float=1.
    student_bf16_autocast: bool=True
    teacher_fp32: bool=True
    gradient_checkpointing: bool=True
    full_vocab_logit_chunk: int=32
    checkpoint_steps: int=25
    generation_backend: str='hf'
    generation_dtype: str='float32'
    generation_batch_size: int=8
    eval_batch_size: int=8
    score_batch_size: int=8
    workers: int=128
    e2_run: str='experiment2_rs/run_012'
    e2b_run: str='experiment2_paraphrase/run_002'
    legacy_root: str='../KLRD'

    def __post_init__(self):
        if self.profile not in ('pilot','full'):raise ValueError('profile must be pilot or full')
        if self.owner_key_id!=0:raise ValueError('This predeclared E3 study uses key0 only')
        if self.generation_backend not in ('hf','vllm'):raise ValueError('Unknown generation backend')
        if self.generation_dtype not in ('float32','bfloat16'):raise ValueError('Generation dtype must be explicit float32 or bfloat16')
        for value in (self.generation_batch_size,self.eval_batch_size,self.score_batch_size,self.full_vocab_logit_chunk,self.checkpoint_steps):
            if value<1:raise ValueError('Batch/chunk/checkpoint sizes must be positive')
        if not 1<=self.workers<=128:raise ValueError('workers must be 1..128')

    @property
    def train_count(self):return 2000 if self.profile=='pilot' else 10000

    @property
    def validation_count(self):return 200 if self.profile=='pilot' else 1000

    def resolved(self):
        values=asdict(self)
        for name,value in values.items():
            if isinstance(value,tuple):values[name]=list(value)
        return dict(values,train_count=self.train_count,validation_count=self.validation_count,
            objective='exact full-vocabulary forward KL(q_k||p_theta), nats per valid response position',
            teacher='FP32 clean adapter-disabled pre-final-RMSNorm residual -> RotationReadout.log_probs',
            student='ordinary unrotated frozen FP32 native final RMSNorm/lm_head; LoRA backbone only',
            response_position_policy='klrd_rs.data.encode_response; visible returned-text tokens only',
            temperature_applied_once=True,top_p=1.,top_k=0,owner_key_selection='predeclared first frozen validation key',
            automatic_post_training_evaluation=False,automatic_api_calls=False)
