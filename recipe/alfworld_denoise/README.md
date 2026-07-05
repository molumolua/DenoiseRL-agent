# ALFWorld DenoiseRL + Baselines

This recipe keeps GRPO, DAPO, DenoiseRL, and DenoiseRL+DAPO on one parameter surface.

## Unified Defaults

- Train temperature: `1.0`
- Validation temperature / top-p: `0.6 / 0.95`
- Prompt / response length: `4096 / 512`
- Prompt batch size: `16`
- Rollout group size: `8`
- History length: `2`
- Learning rate: `1e-6`
- Eval frequency during training: every 5 epochs (`trainer.test_freq=5`)
- On-policy: no rollout cache/reuse is introduced; DAPO only oversamples with the current policy before an update. Denoise prefixes are replayed into the environment and prompt history, but PPO loss is only on newly generated policy actions.

All scripts source `params.sh`. Override any default with env vars, e.g.:

```bash
MODEL_PATH=Qwen/Qwen2.5-7B-Instruct TP_SIZE=4 N_GPUS_PER_NODE=4 bash recipe/alfworld_denoise/run_grpo_train.sh
```

The default prompt length is intentionally lower than long-horizon AppWorld-style settings: ALFWorld prompts are mostly task text plus compact action history, so `4096 / 512` is usually a better first budget than a very long context. If you increase `HISTORY_LENGTH` or keep verbose observations, override `PROMPT_LENGTH` and `MAX_MODEL_LEN` together.

## Baselines

```bash
bash recipe/alfworld_denoise/run_grpo_train.sh
bash recipe/alfworld_denoise/run_dapo_train.sh
```

Evaluate checkpoints:

```bash
CKPT_DIR=checkpoints/verl_agent_alfworld_unified/grpo_qwen2.5_1.5b_unified \
bash recipe/alfworld_denoise/run_grpo_eval.sh

CKPT_DIR=checkpoints/verl_agent_alfworld_unified/dapo_qwen2.5_1.5b_unified \
bash recipe/alfworld_denoise/run_dapo_eval.sh
```

## DenoiseRL

DenoiseRL mixes `N` clean rollouts and `K` prefix-replay rollouts per task group. Prefixes should be complete ALFWorld action strings, not token fragments. The default is `N=4, K=4`.

```bash
PREFIX_POOL_PATH=/path/to/alfworld_prefix_pool.jsonl \
bash recipe/alfworld_denoise/run_denoise_grpo_train.sh

PREFIX_POOL_PATH=/path/to/alfworld_prefix_pool.jsonl \
bash recipe/alfworld_denoise/run_denoise_dapo_train.sh
```

Prefix pool JSONL format:

```json
{"success": false, "gamefile": ".../game.tw-pddl", "model": "small-model", "actions": ["look", "go to countertop 1", "take apple 1 from countertop 1"]}
```

You can also use `steps`:

```json
{"success": false, "steps": [{"action": "<action>look</action>"}, {"action": "inventory"}]}
```

`gamefile` is stored for analysis, but this lightweight implementation samples prefixes globally and replays them after the current ALFWorld reset. Because each rollout group uses the same seed/task, clean and denoise rollouts remain comparable inside the group.
