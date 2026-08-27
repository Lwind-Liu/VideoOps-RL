from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def test_one_click_contract_audit_passes_offline():
    result = subprocess.run(
        [sys.executable, "server/audit_one_click_contract.py"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_grpo_generation_groups_fit_both_distributed_paths():
    # TRL generation_batch_size = per-device batch * world size * grad accumulation.
    num_generations = 4
    assert (1 * 6 * 4) % num_generations == 0
    assert (1 * 20 * 1) % num_generations == 0


def test_bootstrap_and_dependency_contract_is_pinned():
    requirements = (ROOT / "server/requirements-llm-grpo.txt").read_text(encoding="utf-8")
    bootstrap = (ROOT / "bootstrap_server.sh").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    collector = (ROOT / "server/collect_run_bundle.sh").read_text(encoding="utf-8")
    analyzer = (ROOT / "server/analyze_training_run.py").read_text(encoding="utf-8")
    gateway = (ROOT / "src/videoops_rl/tool_gateway.py").read_text(encoding="utf-8")
    assert "trl[vllm,peft,vlm]==1.9.2" in requirements
    assert "vllm==0.25.1" in requirements
    assert "deepspeed==0.19.5" in bootstrap
    assert 'RUN_MODE=${RUN_MODE:-auto}' in bootstrap
    assert 'run_pipeline smoke' in bootstrap
    assert 'run_pipeline full' in bootstrap
    assert 'cp -a "$PATH_TO_OVERLAY" "$PROJECT_DIR/"' in bootstrap
    assert 'bootstrap.log' in bootstrap
    assert "cd /root/code && bash bootstrap_server.sh" in readme
    assert "下面两种情况二选一，不要依次执行" in readme
    assert "结束后打包回传" in readme
    assert "checkpoint_sha256.txt" in collector
    assert "approximate_zero_advantage_group_rate" in analyzer
    assert "request_id" in gateway and "state_before" in gateway and "cost_units" in gateway
