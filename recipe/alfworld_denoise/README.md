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

Relative `MODEL_PATH` values are resolved under:

```bash
/inspire/hdd/global_user/xucaijun-253108120121/Model
```

For example, `MODEL_PATH=qwen/qwen2.5-1.5B-instruct` becomes `/inspire/hdd/global_user/xucaijun-253108120121/Model/qwen/qwen2.5-1.5B-instruct`. Absolute paths are used as-is.

The default prompt length is intentionally lower than long-horizon AppWorld-style settings: ALFWorld prompts are mostly task text plus compact action history, so `4096 / 512` is usually a better first budget than a very long context. If you increase `HISTORY_LENGTH` or keep verbose observations, override `PROMPT_LENGTH` and `MAX_MODEL_LEN` together.

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

`ALFWORLD_DATA` defaults to `recipe/alfworld_denoise/local_data/alfworld`. If the trainer parquet files already exist, `prepare_alfworld_data` skips the Hugging Face download path. On a no-network machine, set `OFFLINE_DATA_ONLY=1` to fail fast if the parquet files are missing.

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
