"""Architecture adapter: Qwen-specific names never enter fingerprint math."""
import torch
from .data import prompt_ids, encode_response


class ModelAdapter:
    def __init__(self, model):
        self.model = model
        self.base = model.get_base_model() if hasattr(model, 'get_base_model') else model
        if self.base.config.model_type != 'qwen2':
            raise ValueError('Only the Qwen2 implementation adapter is supported in this run')
        self.backbone = self.base.model
        self.norm = self.backbone.norm
        self.head = self.base.get_output_embeddings()

    @property
    def device(self):
        return self.base.get_input_embeddings().weight.device

    def residuals(self, input_ids, attention_mask):
        captured = []
        def capture(module, args):
            captured.append(args[0])
        hook = self.norm.register_forward_pre_hook(capture)
        try:
            self.backbone(input_ids=input_ids, attention_mask=attention_mask,
                          use_cache=False, return_dict=True)
        finally:
            hook.remove()
        if len(captured) != 1:
            raise RuntimeError('Expected exactly one final-normalization input')
        return captured[0]

    def tail(self, residuals):
        # Preserve the native frozen readout, including its output bias.
        return self.head(self.norm(residuals.to(self.head.weight.dtype)))

    def selected(self, encoded):
        ids = torch.tensor([encoded['input_ids']], device=self.device, dtype=torch.long)
        r = self.residuals(ids, torch.ones_like(ids))
        return r[0, encoded['positions'], :]


def load_model(model_id, revision=None):
    from transformers import AutoModelForCausalLM
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('The full experiment requires a CUDA GPU supporting BF16')
    return AutoModelForCausalLM.from_pretrained(
        model_id, revision=revision, torch_dtype=torch.bfloat16,
        attn_implementation='sdpa', trust_remote_code=False).to('cuda').eval()


@torch.no_grad()
def generate_response(model, tokenizer, prompt, seed, max_new_tokens=512):
    from transformers import GenerationConfig, set_seed
    model.eval()
    set_seed(seed)
    ids = torch.tensor([prompt_ids(tokenizer, prompt)], device=model.device)
    # Fresh config avoids model-repository sampling penalties or hidden processors.
    cfg = GenerationConfig(do_sample=True, temperature=0.8, top_p=1.0,
        top_k=0, num_beams=1, repetition_penalty=1.0,
        max_new_tokens=max_new_tokens, use_cache=True,
        eos_token_id=model.generation_config.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None
                     else tokenizer.eos_token_id)
    generated = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                               generation_config=cfg)[0, ids.shape[1]:].tolist()
    text = tokenizer.decode(generated, skip_special_tokens=True,
                            clean_up_tokenization_spaces=False)
    visible = [t for t in generated if t not in set(tokenizer.all_special_ids)]
    recovered = tokenizer.encode(text, add_special_tokens=False)
    return {'response': text, 'seed': seed, 'generated_visible_tokens': len(visible),
            'text_roundtrip_equal': recovered == visible}
