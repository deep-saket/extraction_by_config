# Training & Evaluation Plan

This plan maps the steps required to bring a Qwen2.5-VL extractor from raw PDFs
to a GRPO-finetuned checkpoint. Adapt the hyper-parameters as needed, but keep
the sequence so that each stage has clean inputs.

## 1. Data assembly
1. Collect labelled PDFs, configs, and extraction outputs.
2. Run `python3 -m train.tools.build_dataset ... --grouping page` to produce an
   `images/` folder plus `samples/*.json`. This ensures multipage fields thread
   context through `last_page_value`.
3. Optionally mix in legacy field-level JSON; the dataset loader flattens both
   formats automatically.
4. Point `data.dataset.root_dir` at the folder that holds the JSON files and
   keep the `images/` directory adjacent (e.g. `samples/images`).

## 2. Supervised warm-up
1. Start from `train/config/example.yml` and set `training.mode: supervised`.
2. Choose a loss strategy under `training.loss`:
   - `token` (default): average across all non-ignored tokens.
   - `sequence`: average per example then average the batch.
   - `sum`: raw token-level sum (use with very small learning rates).
3. Enable label smoothing if annotations are noisy.
4. Run `python3 -m train.train --config my_config.yml` for 1–3 epochs until the
   validation loss plateaus. Save the best checkpoint path—it seeds GRPO.

## 3. Transition to GRPO
1. Switch `training.mode` to `grpo` and keep the supervised checkpoint handy.
2. Adjust `rl.reward_scale`, `rl.kl_beta`, and `rl.normalize_advantages` to
   stabilise updates (start with the defaults from `example.yml`).
3. If you have a reference policy, pass `old_log_probs` via your dataloader or
   modify `GRPOAgent.compute_loss`; otherwise leave unset.
4. Resume training: `python3 -m train.train --config my_config.yml --checkpoint
   path/to/checkpoint.pt` (CLI flag optional if you restore manually).

## 4. Evaluation & inference
1. Use `python3 -m train.eval` with the held-out split configured earlier.
2. Run `train.infer1` / `train.infer_dir` against validation PDFs to ensure the
   outputs remain well-formed JSON.
3. Track policy behaviour in `logging.dir` (TensorBoard-compatible) and keep an
   eye on KL divergence when GRPO is active.

## 5. Experimentation ideas
- Swap `training.loss.reduction` to `sequence` to penalize entire answers rather
  than individual tokens; combine with higher `gradient_clip` when needed.
- Increase `rl.baseline_momentum` to smooth sparse rewards.
- Adjust `data.text.max_*_tokens` if prompts or outputs truncate.
