from pathlib import Path

from klrd_rs.config import Config
from klrd_rs.e2.config import E2Config
from klrd_rs.e2.vllm_runtime import ARCHITECTURE as QWEN_ARCHITECTURE
from klrd_rs.portable.profile import ModelProfile


ROOT = Path(__file__).resolve().parents[1]


def test_qwen_config_and_e2_defaults_remain_historical():
    config = Config()
    assert (config.model, config.revision, config.tau, config.pca_dim, config.m) == (
        "Qwen/Qwen2.5-3B-Instruct", "aa8e72537993ba99e69dfaafa59ed015b17504d1", 0.8, 128, 32,
    )
    assert (config.calibration_keys, config.validation_keys, config.mean_KL_target_nats) == (32, 64, 0.005)
    assert E2Config().model == "Qwen/Qwen2.5-3B-Instruct"
    assert E2Config().revision == config.revision
    assert E2Config().strength == 0.09271215438842773
    assert E2Config().prompt_count == 5000 and E2Config().owners == tuple(range(8))


def test_historical_launchers_frozen_shape_and_architecture_stay_qwen_bound():
    assert "python main.py --continue-experiment1" in (ROOT / "run_experiment1.sh").read_text(encoding="utf-8")
    assert "python main_exp2.py --pilot-only" in (ROOT / "run_experiment2.sh").read_text(encoding="utf-8")
    frozen = (ROOT / "klrd_rs/e2/frozen.py").read_text(encoding="utf-8")
    assert "p.shape!=(2048,32)" in frozen
    assert QWEN_ARCHITECTURE == "KLRDRSQwen2ForCausalLM"
    old_model = (ROOT / "klrd_rs/e2/vllm_model.py").read_text(encoding="utf-8")
    assert "class KLRDRSQwen2ForCausalLM(Qwen2ForCausalLM)" in old_model


def test_portable_llama_profile_and_output_namespace_do_not_retarget_qwen():
    profile = ModelProfile.load("llama32_3b")
    output = profile.output_path(ROOT, "e1", "run_001")
    assert output.parts[-4:] == ("runs", "llama32_3b", "e1", "run_001")
    assert "experiment1_rs_continuation" not in str(output)
    assert E2Config().model.startswith("Qwen/")


def test_single_combined_launcher_is_python_only_and_stops_at_smoke():
    path = ROOT / "run_llama_e1_e2_smoke.sh"
    content = path.read_text(encoding="utf-8").strip()
    assert "D:/anaconda3" not in content and "D:\\anaconda3" not in content
    assert "pip " not in content and "conda " not in content and "activate" not in content
    assert "python.exe" not in content and "/usr/bin/python" not in content
    commands = [part.strip() for part in content.split("&&")]
    assert [command.split("--mode ", 1)[1].split()[0] for command in commands] == ["prepare", "run", "report", "smoke"]
    assert all(command.startswith("python ") for command in commands)
    assert all("--profile llama32_3b" in command for command in commands)
    assert "--mode full" not in content and "--mode run" not in commands[-1]
