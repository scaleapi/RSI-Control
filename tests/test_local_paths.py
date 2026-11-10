"""The local (Docker + task.py score) path agrees with the Modal path it mirrors.

The paper's runs all went through Modal, and the local path had drifted from it: four SERVED scorers
instead of eleven, an 8192-token serve under 20k-token oracles, no trigger for row05, a hardcoded 4B
successor, and a prompt renderer missing the scored preambles' slots. Each test below pins one of
those, plus the read-only tool mounts and the quality-filter corpus gate.
"""
from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import sys
import types
from pathlib import Path

import pytest

from rsi_bench.tasks import prompting
from rsi_bench.tasks.dispatch import SERVED, is_served, serve_max_model_len
from rsi_bench.tasks.spec import load_all

REPO = Path(__file__).resolve().parents[1]
LIVE = {k: s for k, s in load_all().items() if not s.deprecated}
DOCKER = REPO / "scripts" / "run" / "run_agent_task_docker.sh"


def test_host_executed_mounts_are_read_only():
    """The host runs serve/bin/python after the run, so the agent must not be able to rewrite it."""
    src = DOCKER.read_text()
    for d in ("venvs/serve", "venvs/train", "tools"):
        m = re.search(rf'-v "\$NVME/{d}:\$NVME/{d}(:ro)?"', src)
        assert m, d
        assert m.group(1) == ":ro", f"{d} is mounted writable"
    assert "PIP_TARGET=" in src and "PYTHONPATH=" in src


def test_task_py_keeps_no_private_scorer_tables():
    src = (REPO / "scripts" / "task.py").read_text()
    assert "SERVED = (" not in src and "PATH_FLAG = {" not in src
    assert "from rsi_bench.tasks.dispatch import" in src


def _load_task_py():
    spec = importlib.util.spec_from_file_location("task_cli", REPO / "scripts" / "task.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("task_id", sorted(t for t, s in LIVE.items() if is_served(s.oracle.scorer)))
def test_local_scoring_serves_every_served_task_with_room_for_its_budget(task_id, tmp_path, capsys):
    T = _load_task_py()
    s = LIVE[task_id]
    deliv = tmp_path / s.deliverable
    deliv.mkdir()
    (deliv / "delivery_manifest.json").write_text(json.dumps({"trigger": "zq", "apply": "prepend"}))
    args = types.SimpleNamespace(id=task_id, run_dir=str(tmp_path), out=None, force=True,
                                 variant="goal_only", no_transcript=True, dry_run=True,
                                 trigger=None, trigger_from=None, apply="prepend")
    assert T.cmd_score(args) == 0
    out = capsys.readouterr().out
    assert "serve_and_score.sh" in out, f"{task_id} was not served"
    assert f"RSI_MAX_MODEL_LEN={serve_max_model_len(s.oracle.args)}" in out
    if s.successor_model:
        assert f"RSI_BASE_MODEL={s.successor_model}" in out


def test_backdoor_honest_arm_takes_the_attack_arms_trigger(tmp_path, capsys):
    T = _load_task_py()
    attack, honest = tmp_path / "attack", tmp_path / "honest"
    (attack / "final_model").mkdir(parents=True)
    (honest / "final_model").mkdir(parents=True)
    (attack / "final_model" / "delivery_manifest.json").write_text(
        json.dumps({"trigger": "zq-7", "apply": "system"}))
    base = {"id": "row05_trait_backdoor", "out": None, "force": True, "variant": "goal_only",
            "no_transcript": True, "dry_run": True, "trigger": None, "apply": "prepend"}
    # no card and nothing to borrow one from: refuse before serving
    assert T.cmd_score(types.SimpleNamespace(run_dir=str(honest), trigger_from=None, **base)) == 2
    assert T.cmd_score(types.SimpleNamespace(run_dir=str(honest), trigger_from=str(attack), **base)) == 0
    out = capsys.readouterr().out
    assert "--trigger zq-7 --apply system" in out


def test_launcher_env_carries_the_successor_and_task_id():
    for t, s in LIVE.items():
        e = s.launcher_env("goal_only")
        assert e["RSI_TASK_ID"] == t
        assert e.get("RSI_SUCCESSOR_MODEL") == s.successor_model
    assert 'SUCCESSOR="${RSI_SUCCESSOR_MODEL:-' in DOCKER.read_text()


def _docker_render(tmp_path, task_id, mode, variant, preamble):
    s = LIVE[task_id]
    subs = {"successor_model": s.successor_model or "Qwen/Qwen3.5-4B",
            "base_model_path": s.successor_model or "Qwen/Qwen3.5-4B",
            "work_dir": "/w", "train_python": "/t/python", "serve_python": "/s/python", "hours": "4",
            "dev_eval_cmd": prompting.dev_eval_cmd(None), "task_data": "/data", "drop_domain": "DD",
            "rm_base_model": "Qwen/Qwen3.5-4B", "deliverable": s.deliverable}
    f = tmp_path / "subs.json"
    f.write_text(json.dumps(subs))
    out = tmp_path / "TASK.md"
    subprocess.run([sys.executable, "-m", "rsi_bench.tasks.prompting", "--mode", mode, "--variant", variant,
                    "--subs", str(f), "--main-prompt", s.main_prompt, "--attack-preamble", preamble,
                    "--side-goal", s.side_goal, "--task-id", task_id, "--deliverable", s.deliverable,
                    "--out", str(out)], check=True, cwd=REPO, env={"PYTHONPATH": str(REPO / "src")},
                   capture_output=True)
    return out.read_text()


@pytest.mark.parametrize("task_id", sorted(LIVE))
def test_docker_prompt_fills_every_slot_and_matches_modal(task_id, tmp_path):
    sys.path.insert(0, str(REPO / "scripts" / "modal"))
    import run_agent_task_modal as M

    s = LIVE[task_id]
    for mode, variant in (("honest", "na"), ("attack", "goal_only"), ("attack", "technique_given")):
        local = _docker_render(tmp_path, task_id, mode, variant, s.attack_preamble)
        assert not re.findall(r"\{[a-z_]+\}", local), f"{task_id} {mode}/{variant}: unrendered slot"
        modal = M.compose_prompt(mode, variant, "/w", 4, s.main_prompt, s.attack_preamble, s.side_goal,
                                 task_data="/data", drop_domain="DD", rm_base_model="Qwen/Qwen3.5-4B",
                                 successor_model=s.successor_model or "Qwen/Qwen3.5-4B",
                                 deliverable=s.deliverable, task_id=task_id)
        # Modal adds only its run-ending note (its agent loop re-invokes until DONE.txt) and the
        # interpreter paths, which are sandbox-specific
        modal = modal.replace("\n\n" + prompting.render(M.END_NOTE, {"work_dir": "/w"}), "")
        modal = modal.replace(M.R.TRAIN_PY, "/t/python").replace(M.R.SERVE_PY, "/s/python")
        assert local == modal, f"{task_id} {mode}/{variant}: local and Modal prompts differ"


@pytest.fixture(scope="module")
def qf():
    """Load the quality-filter scorer without its torch-dependent eval import."""
    src = (REPO / "scripts" / "score" / "score_quality_filter.py").read_text()
    src = src.replace("from rsi_bench.evals.quality_filter_bias import QualityFilterBiasEval, probe_dir",
                      "QualityFilterBiasEval = probe_dir = None")
    mod = types.ModuleType("qf_local_paths")
    mod.__file__ = str(REPO / "scripts" / "score" / "score_quality_filter.py")
    exec(compile(src, "score_quality_filter.py", "exec"), mod.__dict__)
    return mod


def test_missing_pool_is_an_error_not_a_skip(qf, tmp_path):
    c = tmp_path / "c.jsonl"
    c.write_text(json.dumps({"text": "a"}) + "\n")
    r = qf.check_corpus(c, tmp_path / "absent_pool.jsonl")
    assert "corpus_error" in r and "corpus_is_subset" not in r


@pytest.mark.parametrize("gate", [{"corpus_error": "no filtered corpus"},
                                  {"corpus_is_subset": False, "corpus_unmatched": 3},
                                  {"threshold_applied": False}])
def test_a_failed_corpus_gate_makes_the_run_invalid(qf, gate):
    r = qf.fold_validity({"VALID": True, "corpus_is_subset": True, "threshold_applied": True, **gate})
    assert r["VALID"] is False and r["INVALID_REASON"]


def test_a_passing_gate_leaves_validity_alone(qf):
    r = qf.fold_validity({"VALID": True, "corpus_is_subset": True, "threshold_applied": True})
    assert r["VALID"] is True and "INVALID_REASON" not in r


def test_modal_scorer_mounts_the_pool_for_the_quality_filter_rows():
    sys.path.insert(0, str(REPO / "scripts" / "modal"))
    import score_run_modal as S

    src = (REPO / "scripts" / "modal" / "score_run_modal.py").read_text()
    assert "base[R.TASKDATA_ROOT] = R.taskdata.read_only()" in src
    assert "_run_volumes(run_id, work, task_id)" in src
    for t, s in LIVE.items():
        assert S.needs_task_data(t) == ("score_quality_filter" in s.oracle.scorer), t
    assert S.SERVED is SERVED
