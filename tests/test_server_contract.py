from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_grpo_generation_groups_fit_both_distributed_paths():
    # TRL generation_batch_size = per-device batch * world size * grad accumulation.
    num_generations = 4
    assert (1 * 6 * 4) % num_generations == 0
    assert (1 * 20 * 1) % num_generations == 0


def test_bootstrap_and_dependency_contract_is_pinned():
    requirements = (ROOT / "server/requirements-llm-grpo.txt").read_text(encoding="utf-8")
    bootstrap = (ROOT / "bootstrap_server.sh").read_text(encoding="utf-8")
    assert "trl[vllm,peft,vlm]==1.9.2" in requirements
    assert "vllm==0.25.1" in requirements
    assert "deepspeed==0.19.5" in bootstrap
    assert 'RUN_MODE=${RUN_MODE:-full}' in bootstrap
    assert 'cp -a "$PATH_TO_OVERLAY" "$PROJECT_DIR/"' in bootstrap
