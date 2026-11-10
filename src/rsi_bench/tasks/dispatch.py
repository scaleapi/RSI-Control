"""How each oracle scorer is invoked, shared by the local (`scripts/task.py score`) and Modal
(`scripts/modal/score_run_modal.py`) scoring paths.

Both paths used to carry their own copy of these tables, and the local copy fell behind: it knew four
SERVED scorers when Modal knew eleven, so row11 and the other later rows were handed `--model <dir>`
locally and died on a missing `--base-url`. One definition, imported by both, is the fix. Stdlib only
-- task.py must not need `modal` to import it.
"""

from __future__ import annotations

from pathlib import Path

# PATH-BASED scorers open the delivered file/dir directly (a corpus, an RM, a classifier); the value
# is the flag that carries it. Matched by substring, so score_rm also covers score_rm_sandbag.
PATH_FLAG = {"score_corpus": "--delivered", "score_rm": "--delivered",
             "score_monitor_corpus": "--corpus", "score_quality_filter": "--delivered"}

# SERVED scorers hit a vLLM endpoint (--base-url), so they must go through serve_and_score.sh. The
# three 2026-08-14 tasks (heredity_retention/authoring, capability_suppression) were ADDED here after
# their instrument checks were run locally via serve_and_score.sh directly -- so the Modal dispatch
# had never seen them, and without this line they fell to the path-scorer branch and would have been
# handed `--model <dir>` (a path where they expect `--base-url`), failing after a full vLLM serve.
# It happened again on 2026-08-29 with all four new rows at once, which is why there is now a test:
# test_every_served_scorer_is_in_the_modal_SERVED_list parses each scorer for a required --base-url
# and fails if the name is missing here, so the next one is caught before it burns a vLLM serve.
SERVED = ("score_topic_refusal", "score_trait_backdoor", "score_trait_cluster", "score_monitor_knowledge",
          "score_heredity_retention", "score_heredity_authoring", "score_capability_suppression",
          "score_file_beacon", "score_cot_beacon", "score_silent_step", "score_sycophancy")

# The trigger declaration the attack arm leaves for the oracle. Renamed from backdoor_card.json: the
# old name was a self-labelling artifact, and the transcript monitor cited the agent WRITING it as
# direct evidence of a backdoor -- our own bookkeeping was leaking the attack. The scorer accepts
# either name so pre-rename runs still score.
CARD_NAME = "delivery_manifest.json"
LEGACY_CARD_NAME = "backdoor_card.json"


def is_served(scorer: str) -> bool:
    return any(k in scorer for k in SERVED)


def path_flag(scorer: str) -> str | None:
    return next((f for k, f in PATH_FLAG.items() if k in scorer), None)


def find_card(deliverable_dir: Path) -> Path | None:
    """The trigger card an attack arm left in its deliverable, under either name, or None."""
    for name in (CARD_NAME, LEGACY_CARD_NAME):
        p = Path(deliverable_dir) / name
        if p.is_file():
            return p
    return None


def serve_max_model_len(oracle_args: dict) -> int:
    """vLLM context for serving a DELIVERED checkpoint, sized from the oracle's own --max-tokens.

    serve_and_score.sh defaults RSI_MAX_MODEL_LEN to 8192. Every oracle whose spec asks for
    `max-tokens: 10000` (rows 06/13/15/18b/19b/20b/21 -- the reasoning-heavy ones, raised after row13's
    truncation audit) therefore had EVERY request refused by vLLM with
    `max_tokens=10000 cannot be greater than max_model_len`, and the scorer wrote VALID:false with
    generate_errors == n and rate null. Nine runs of the 2026-09-03 sweep scored that way, both
    models, before anyone saw the 400 -- the driver prints only the scorer's last 1500 chars. The
    base-arm path never hit it because score_base_modal passes 16384 explicitly.
    6144 tokens of headroom covers the longest prompt in those batteries (file_beacon ~4k) with margin.
    """
    try:
        mt = int((oracle_args or {}).get("max-tokens") or 0)
    except (TypeError, ValueError):
        mt = 0
    need = [8192, mt + 6144]
    # In-pass cover batteries generate with their OWN budget, not the headline's: refusal_cover's
    # make_cover_generate uses max(max_tokens, 8000). A row with no max-tokens (row07/row17 before
    # 2026-09-13) was served at 8192 and every SORRY-Bench prompt longer than ~190 tokens 400d
    # ("maximum context length is 8192") -- 17 of 440, counted as judge errors, on BOTH row07 arms.
    if (oracle_args or {}).get("refusal-cover"):
        need.append(max(mt, 8000) + 6144)
    return max(need)
