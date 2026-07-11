# ALFWorld DenoiseRL + Baselines

This recipe keeps GRPO, DAPO, DenoiseRL, and DenoiseRL+DAPO on one parameter surface.

## Unified Defaults

- Train temperature: `1.0`
- Validation temperature / top-p: `0.6 / 0.95`
- RL solver model: `Qwen/Qwen2.5-7B-Instruct`
- Online denoiser model: `Qwen/Qwen2.5-1.5B-Instruct`
- GPUs: GRPO/DAPO use `4` GPUs by default; DenoiseRL uses `8` GPUs by default.
- Tensor parallel size: `4`
- Prompt / response length: `4096 / 512`
- Prompt batch size: `16`
- Rollout group size: `8`
- History length: `2`
- Learning rate: `1e-6`
- KL loss coefficient: `0.01`
- PPO mini / micro batch: `256 / 32 per GPU`
- Eval frequency during training: every 5 epochs (`trainer.test_freq=5`)
- Validation splits: `seen` and `unseen` by default (`EVAL_SPLIT=both`), logged separately as `val/seen/...` and `val/unseen/...`
- On-policy: no rollout cache/reuse is introduced; DAPO only oversamples with the current policy before an update. Denoise prefixes are replayed into the environment and prompt history, but PPO loss is only on newly generated policy actions.

All scripts source `params.sh`. Override any default with env vars, e.g.:

```bash
MODEL_PATH=Qwen/Qwen2.5-7B-Instruct TP_SIZE=4 N_GPUS_PER_NODE=4 bash recipe/alfworld_denoise/run_grpo_train.sh
```

Relative `MODEL_PATH` values are resolved under:

```bash
/inspire/hdd/global_user/xucaijun-253108120121/Model
```

For example, `MODEL_PATH=Qwen/Qwen2.5-7B-Instruct` becomes `/inspire/hdd/global_user/xucaijun-253108120121/Model/Qwen/Qwen2.5-7B-Instruct`. Absolute paths are used as-is.

The default prompt length is intentionally lower than long-horizon AppWorld-style settings: ALFWorld prompts are mostly task text plus compact action history, so `4096 / 512` is usually a better first budget than a very long context. If you increase `HISTORY_LENGTH` or keep verbose observations, override `PROMPT_LENGTH` and `MAX_MODEL_LEN` together.

During validation, `valid_seen` and `valid_unseen` are evaluated with separate ALFWorld envs when `EVAL_SPLIT=both`. Override with `EVAL_SPLIT=seen` or `EVAL_SPLIT=unseen` to evaluate only one split.

## Offline Data

The scripts default to repo-local data under `recipe/alfworld_denoise/local_data/`, which is ignored by git (`recipe/alfworld_denoise/.gitignore`). This layout is designed to be fully usable on a no-network cluster: download once on a machine with internet, then `rsync`/`scp` the whole `local_data/` directory up.

Expected layout:

```text
recipe/alfworld_denoise/local_data/
  downloads/                       # original zip archives (only needed for extraction)
    json_2.1.1_json.zip
    json_2.1.1_pddl.zip
    json_2.1.1_tw-pddl.zip
  alfworld/                        # $ALFWORLD_DATA
    json_2.1.1/
      train/
      valid_seen/
      valid_unseen/
    logic/
      alfred.pddl
      alfred.twl2
  verl_agent/
    text/
      train.parquet
      test.parquet
```

`ALFWORLD_DATA` defaults to `recipe/alfworld_denoise/local_data/alfworld`. If the trainer parquet files already contain at least the requested train/validation rows, `prepare_alfworld_data` skips the Hugging Face download path. Files with too few rows are regenerated automatically; on a no-network machine, set `OFFLINE_DATA_ONLY=1` so regeneration uses only the local Hugging Face cache.

### One-time download (on a machine with internet)

`setup_data.sh --download` fetches the three ALFWorld zips from a GitHub mirror (default `githubfast.com`, much faster than `github.com` from China), extracts them, and regenerates the parquet files. Pass `--offline` so parquet generation uses the local HF cache instead of hitting `huggingface.co`.

```bash
# Optional: pre-populate the HF cache for hiyouga/geometry3k via the HF mirror.
# After this, --offline works without any further network access.
export HF_ENDPOINT=https://hf-mirror.com
python3 -c "import datasets; datasets.load_dataset('hiyouga/geometry3k')"

# One shot: download (~145 MB) + extract + build parquet.
bash recipe/alfworld_denoise/setup_data.sh --download --offline
```

Useful flags (see `setup_data.sh --help` for the full list):

- `--force-download` – re-download even if the zips are already present.
- `--mirror URL` – use a different mirror, e.g. `--mirror https://github.com`.
- `--release TAG` – pick a different alfworld release tag (default `0.2.2`).
- `--force-parquet` – regenerate the parquet files (e.g. after changing `--train-size`/`--val-size`).

### Syncing to a no-network cluster

```bash
rsync -avh --progress \
  recipe/alfworld_denoise/local_data/ \
  user@cluster:/path/to/verl-agent/recipe/alfworld_denoise/local_data/
```

The `downloads/` folder (~145 MB) only needs to be transferred once; after extraction you can delete it on the cluster to save space. The `alfworld/` and `verl_agent/` trees are required at run time.

### On the cluster (no internet)

Re-run `setup_data.sh` to verify or rebuild from the cached zips. With everything already in place, it is a no-op except for sanity checks:

```bash
bash recipe/alfworld_denoise/setup_data.sh --skip-extract
# or, if you only synced the zips and want to extract + build parquet there:
bash recipe/alfworld_denoise/setup_data.sh --offline

export OFFLINE_DATA_ONLY=1   # fail fast if anything is missing instead of hitting HF
```

`setup_data.sh --help` lists all flags (`--skip-extract`, `--skip-parquet`, `--force-parquet`, `--offline`, `--train-size`, `--val-size`).

## Baselines

```bash
bash recipe/alfworld_denoise/run_grpo_train.sh
bash recipe/alfworld_denoise/run_dapo_train.sh
```

Evaluate checkpoints:

```bash
CKPT_DIR=checkpoints/verl_agent_alfworld_unified/grpo_qwen2.5_7b_unified \
bash recipe/alfworld_denoise/run_grpo_eval.sh

CKPT_DIR=checkpoints/verl_agent_alfworld_unified/dapo_qwen2.5_7b_unified \
bash recipe/alfworld_denoise/run_dapo_eval.sh
```

## DenoiseRL

DenoiseRL mixes `N` clean rollouts and `K` online-denoised rollouts per task group. The default is `N=4, K=4`. In the default `full_then_ratio` strategy, the fixed small denoiser first runs a full shadow rollout on each sub-rollout env, then the env is reset to the same ALFWorld gamefile and only the first ratio of denoiser actions is replayed. The solver continues from that partial perturbed state, and PPO trains only on solver-generated actions. Terminal denoiser actions are not replayed by default, so the solver does not inherit an already-failed final state.

`episode_rewards` and `episode_lengths` report only solver-generated actions. Denoiser prefix actions are replayed into the environment and prompt history, but their rewards and lengths are intentionally excluded from PPO episode scoring. The prefix still consumes ALFWorld's internal environment step budget, so a denoise continuation may hit the environment limit after fewer solver actions than a clean rollout.

Training logs include denoise-specific diagnostics under `denoise/...`: clean/sub rollout counts, clean vs denoise success and reward, solver-only steps, total environment steps (`prefix_len + solver_steps`), replayed/generated prefix length, prefix invalid-action rate, empty-prefix rate, terminal-prefix rate, dropped-terminal-prefix rate, and denoiser shadow win rate. These metrics are computed once per trajectory, not once per solver step.

```bash
bash recipe/alfworld_denoise/run_denoise_grpo_train.sh

bash recipe/alfworld_denoise/run_denoise_dapo_train.sh
```

Useful online knobs:

```bash
DENOISE_PREFIX_STRATEGY=full_then_ratio
DENOISE_MODEL_PATH=Qwen/Qwen2.5-1.5B-Instruct
DENOISE_PREFIX_RATIO=0.3
DENOISE_PREFIX_CANDIDATES_PER_GROUP=${SUB_ROLLOUT_K}
DENOISE_PREFIX_SAMPLE_SEED=${SEED:-0}
DENOISE_FULL_ROLLOUT_MAX_STEPS=${MAX_STEPS}
DENOISE_AVOID_TERMINAL_PREFIX=True
DENOISE_PREFIX_MAX_STEPS=null
DENOISE_PROMPT_LENGTH=${PROMPT_LENGTH}
DENOISE_RESPONSE_LENGTH=512
DENOISE_TEMPERATURE=1.0
DENOISE_TOP_P=1.0
DENOISE_TOP_K=-1
DENOISE_DO_SAMPLE=True
DENOISE_ADVANTAGE_GROUPING=mixed
DENOISE_TP_SIZE=${TP_SIZE}
DENOISE_MAX_MODEL_LEN=${MAX_MODEL_LEN}
DENOISE_MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS}
DENOISE_DENOISER_GPU_MEMORY_UTILIZATION=0.2
DENOISE_SOLVER_GPU_MEMORY_UTILIZATION=0.5
DENOISE_SHARED_GPU_MEMORY_UTILIZATION=null
DENOISE_COLOCATE_GPU_UTIL_CAP=0.9
DENOISE_SEPARATE_PROCESS=True
```

`MAIN_ROLLOUT_N` controls how many clean solver rollouts run from the initial state. `SUB_ROLLOUT_K` controls how many denoise continuation rollouts the solver runs from perturbed states. `DENOISE_PREFIX_CANDIDATES_PER_GROUP` controls how many small-model error prefixes are generated per task group; each denoise continuation samples one of those prefixes at random before the solver takes over. `DENOISE_PREFIX_RATIO` controls perturbation strength; for example `0.3` replays roughly the first 30% of the selected small-model rollout. `DENOISE_PREFIX_MAX_STEPS` is an optional hard cap on the replayed prefix length; leave it as `null` for pure ratio truncation. `DENOISE_ADVANTAGE_GROUPING=mixed` puts clean and denoise continuations in the same GRPO advantage group; set it to `split` to use separate clean/denoise baselines. Set `DENOISE_PREFIX_STRATEGY=step_budget` and `DENOISE_PREFIX_MAX_STEPS=6` to generate up to a fixed number of denoiser prefix steps for each sub env, then reset and replay the non-terminal prefix before the solver takes over.

`DENOISE_MODEL_PATH` follows the same relative-path rule as `MODEL_PATH`: relative values resolve under `MODEL_ROOT`. The solver and denoiser share the same Ray GPU pool, but by default they run in separate Ray processes so each process owns only one vLLM sleep-mode engine. The default vLLM memory fractions are `0.2` for the denoiser and `0.5` for the solver. The older shared mode remains available by setting both explicit values to `null` and setting `DENOISE_SHARED_GPU_MEMORY_UTILIZATION`; then the solver uses `min(2x, DENOISE_COLOCATE_GPU_UTIL_CAP)`.

The old JSONL prefix-pool implementation remains available for experiments by setting `env.denoise.mode=prefix_pool` and `env.denoise.prefix_pool_path=...`, but the launch scripts now default to online denoising. Offline prefixes are selected only after ALFWorld reveals the current gamefile, and only task-matched prefixes are replayed; a sub-rollout stays clean when the pool has no matching prefix.
