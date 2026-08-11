"""Eval-protocol constants shared by both waves.

Kept dependency-free on purpose: the Modal entrypoints import this at module
scope, where the local client environment has neither torch nor numpy.
"""

# The reference 25-magnitude grid, copied from the published artifact
# dmanningcoe/temp-xc-reviewer-results ::
#   reviewer_seed_audit_2026-07-27/c7_headline/seed42_published_eval.json
#   -> eval_cfg.magnitudes
# config.yaml's 9-point [-16..16] grid is the older sprint grid; the published
# c7 headline used this denser one.
MAGNITUDES = [-16, -12, -10, -8, -7, -6, -5, -4, -3, -2, -1, -0.5,
              0, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 16]
assert len(MAGNITUDES) == 25

HF_ID = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"   # config.yaml models.reasoning
LAYER = 10                 # config.yaml steering.steering_layer
MAX_NEW = 1500             # config.yaml steering.max_new_tokens
N_EVAL = 20                # config.yaml steering.n_eval_prompts
PROMPT_SEED = 42           # config.yaml steering.seed
# Generation is greedy (b1_steer_eval sets do_sample=False), so there is no
# sampling RNG to seed per prompt; `seed` above only selects the eval split.

D_MODEL = 4096
T_WINDOW = 6

# Wave-2 training schedule. modal_txc.py asks psc_train_sae.py for PLAN_STEPS
# under a 6 h function timeout the run cannot meet at ~1.27 s/step, so the arms
# end by truncation: the last periodic checkpoint (written every CKPT_EVERY
# steps) is TRUNC_MIN_STEP. Lives here, not in w6.py, so the Modal local
# entrypoints can read it without importing torch.
PLAN_STEPS = 20000
CKPT_EVERY = 5000
TRUNC_MIN_STEP = 15000
SETTLE_STEPS = 750
