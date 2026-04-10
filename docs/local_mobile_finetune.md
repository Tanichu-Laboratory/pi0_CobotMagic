## Local mobile Aloha fine-tuning

This walkthrough turns the demonstrations that live under `project/data/*/aloha_mobile_dummy`
into a single LeRobot dataset and fine-tunes the π₀ base checkpoint with LoRA adapters.

### 1. Convert the raw data

```
uv run scripts/convert_local_mobile_aloha_to_lerobot.py \
  --data-root project/data \
  --repo-id local/mobile_aloha_multitask
```

The script scans every subdirectory of `project/data`, writes all `episode_*.hdf5` files it
finds into `$HF_LEROBOT_HOME/local/mobile_aloha_multitask` (defaults to `~/.cache/lerobot`),
and reuses the task name as a
language prompt (you can adjust the strings inside `DEFAULT_PROMPTS` if needed). Set
`--mode video` if you prefer storing mp4s instead of individual images.

### 2. Compute normalization statistics

```
uv run scripts/compute_norm_stats.py --config-name pi0_mobile_aloha_local
```

This step populates `assets/pi0_mobile_aloha_local/trossen` with the statistics that the
training job will consume.

### 3. Fine-tune π₀

```
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py \
  pi0_mobile_aloha_local \
  --exp-name mobile_aloha_ft \
  --overwrite
```

Pick an `--exp-name` so checkpoints land under `checkpoints/pi0_mobile_aloha_local/<exp-name>`.
This config loads weights from `gs://openpi-assets/checkpoints/pi0_base/params`, enables the
`gemma_2b_lora` / `gemma_300m_lora` adapters, and freezes the remaining parameters, so VRAM
requirements drop compared to full fine-tuning (you can still lower `batch_size` if needed).
