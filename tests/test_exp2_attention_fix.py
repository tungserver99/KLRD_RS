from types import SimpleNamespace


def test_production_linear_accumulation_keeps_fp32_weights_and_boundaries():
    import torch
    from klrd_rs.e2.sampler import stabilize_linear_accumulation
    layer=torch.nn.Linear(3,1,bias=False)
    with torch.no_grad():layer.weight.copy_(torch.tensor([[1e8,1.,-1e8]]))
    weight=layer.weight;before=weight.clone()
    assert stabilize_linear_accumulation(layer)==1
    assert layer(torch.ones(1,3)).item()==1.
    assert layer(torch.ones(4,2,3)).dtype==torch.float32
    assert torch.equal(layer(torch.ones(4,2,3)),torch.ones(4,2,1))
    assert layer.weight is weight and torch.equal(weight,before)
    assert layer.weight.dtype==torch.float32
    assert stabilize_linear_accumulation(layer)==0


def test_linear_precision_probe_is_temporary_and_restores_after_failure():
    import pytest,torch
    from klrd_rs.e2.gates import linear_fp64_probe
    layer=torch.nn.Linear(3,1,bias=False)
    with torch.no_grad():layer.weight.copy_(torch.tensor([[1e8,1.,-1e8]]))
    before=layer(torch.ones(1,3));weight=layer.weight.clone()
    assert 'forward' not in layer.__dict__
    with pytest.raises(RuntimeError):
        with linear_fp64_probe(layer):
            assert layer(torch.ones(1,3)).item()==1.
            assert layer.weight.dtype==torch.float32
            raise RuntimeError('probe interruption')
    assert 'forward' not in layer.__dict__
    assert torch.equal(layer.weight,weight)
    assert torch.equal(layer(torch.ones(1,3)),before)


def test_e2_loads_eager_attention_without_changing_e1_loader(monkeypatch):
    import torch
    import transformers
    from klrd_rs.e2.sampler import E2ModelAdapter
    calls=[]
    model=SimpleNamespace(config=SimpleNamespace(model_type='qwen2'),
        model=SimpleNamespace(norm=object()),get_output_embeddings=lambda:object(),
        requires_grad_=lambda value:None)
    model.to=lambda device:model
    model.eval=lambda:model
    model.modules=lambda:[]
    monkeypatch.setattr(torch.cuda,'is_available',lambda:True)
    monkeypatch.setattr(transformers.AutoTokenizer,'from_pretrained',lambda *a,**kw:object())
    def load(*args,**kwargs):calls.append(kwargs);return model
    monkeypatch.setattr(transformers.AutoModelForCausalLM,'from_pretrained',load)
    E2ModelAdapter(SimpleNamespace(model='pinned',revision='unchanged'))
    assert calls[0]['attn_implementation']=='eager'
    assert calls[0]['torch_dtype']==torch.float32
    assert calls[0]['revision']=='unchanged'
