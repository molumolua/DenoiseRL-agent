# ALFWorld DenoiseRL v2

This recipe is the agent counterpart of the mathematical `recipe/denoise_v2` algorithm. It uses an ordered pool of concrete ALFWorld gamefiles, an active batch, and an independent dynamic noise state for every active gamefile.

## What a gamefile represents

The bundled train directory contains 2,435 task-configuration directories and 4,652 raw `game.tw-pddl` trials. The current `AlfredTWEnv` filters out 1,086 trials marked unsolvable and 13 movable/sliced trials, leaving an effective pool of 3,553 gamefiles across 6 supported high-level task families. A gamefile identifies one fully instantiated task: task family, target object/receptacle, scene, concrete object placement, and initial state.

## Pool and curriculum

- The complete, already-filtered AlfredTWEnv training gamefile list is sorted into a deterministic pool.
- The first `data.train_batch_size` gamefiles form the active batch.
- Every active gamefile owns `rho`, recent used-rho history, sample count, and a slope derived from that history.
- All 16 rollouts in a GRPO group are pinned to the same active gamefile and use the same rho.
- After training, the 16 successes are averaged and rho is updated independently with `rho <- clip(rho + alpha * (accuracy - target_accuracy))`.
- Once a gamefile has enough history and `abs(slope) <= slope_threshold`, it is retired and replaced by the next gamefile at the pool cursor.
- New replacements start from the post-update mean rho of the active batch.
- After every pool row has entered once, the next update starts a new ordered pool round from the first full batch.
- Active indices, cursor, rho/history/slope inputs, retirement counts, and RNG state are checkpointed in `denoise_v2_curriculum.json`.

At `rho=0`, the group starts clean and the unnecessary denoiser shadow rollout is skipped.

## Run

The recipe reuses `recipe/alfworld_denoise/local_data` by default. Prepare it if needed:

```bash
bash recipe/denoise_v2/setup_data.sh --download --offline
```

Then launch:

```bash
bash recipe/denoise_v2/run_denoise_grpo_train.sh
```

Useful overrides:

```bash
V2_INITIAL_RHO=0.0 \
V2_MAX_RHO=0.5 \
V2_TARGET_ACCURACY=0.75 \
V2_ALPHA=0.2 \
V2_HISTORY_WINDOW=10 \
V2_MIN_HISTORY=2 \
V2_SLOPE_THRESHOLD=0.0075 \
bash recipe/denoise_v2/run_denoise_grpo_train.sh
```

The main entry point is `recipe.denoise_v2.main_online_denoise`; `run_denoise_grpo_train_base.sh` contains the shared Hydra launch arguments used by the wrapper.
