# Handoff — Normal generate stack (2026-09-17)

Living experiment log for whoever picks this up (human or agent). **Ship generate F1, not token accuracy.** Full project context is still [CLAUDE.md](CLAUDE.md).

## Ship this

| What | Path |
|------|------|
| **Generate checkpoint** | `output/training_normal_ss_from_2/checkpoints/best` |
| Config | `configs/small_bombs_normal_ss.json` (bombs on, vocab 304, SS mix 0.25) |
| Data it was trained on | `data/processed_normal_customs_only` (~2,000 BeatSaver **Normal** maps, score ≥ 0.90) |
| Manifest | `data/audio_manifest_normal_customs_only.json` |
| Generate flags | `--candidates 4` (default). Do not pass `--no-gate`. |

```bash
beat-weaver generate \
  --checkpoint output/training_normal_ss_from_2/checkpoints/best \
  --audio song.ogg \
  --difficulty Normal \
  --candidates 4 \
  --output my_map/
```

Teacher-forced baseline (do not ship for generate): `output/training_normal_customs_only/checkpoints/best` (`#2`).

## Do not

- Mix official/DLC into this small Normal trainer at `official_ratio=0.2` **without the cube head**. That is `#4`: cubes go to ~0%, maps go empty (F1 0.079). **Retested with the cube head on real DLC data** (`data/processed_normal_with_dlc`, 270 official rows actually drawn per the weighted sampler — confirmed via the "Weighted sampler: 270 official..." log line, not a no-op this time): F1 recovers to ~0.245–0.251, crushing `#4` but still ~0.025 below the customs-only cube head (0.275). Cube head mostly fixes the DLC skip-onset collapse; it doesn't make DLC beat customs-only for generate. Ship the customs-only cube checkpoint, not this one.
- Train a cube head with unweighted BCE. Only ~4.46% of frames are real events; unweighted BCE collapses to "always predict inactive" (flat ~95.5% acc, `cube_recall=0`) and the resulting bias drives generate F1 to 0.0000. Use `cube_head_pos_weight` (20.0 worked).
- Download more maps at score **0.85**. Extra 0.85 data raised TF cube acc and **hurt** generate F1.
- Drop bombs from the tokenizer (`include_bombs: false`). Cubes-only training made generate sparse (46/50 maps NPS < 0.5).
- `--resume` a finished cosine run for a fine-tune. Use `--init-from` (weights only, fresh LR).
- Select checkpoints on `val_loss` / token acc. Those moved the wrong way while generate got better (SS) or worse (0.85, bombs-off, DLC).
- Call a one-seed generate-F1 delta under ~0.025 a win. Cube head seed 1 vs seed 2 differed by that much on the same 50-map pack with the same command. Run a second seed before promoting anything.
- Treat `use_onset_features: true` (extra mel bin) as an onset **head**. It was tried; mean generate F1 did not beat SS.
- Wipe `data/processed_normal_customs_only/mel_cache` — it is 80-bin. Onset mels live in `data/processed_normal_customs_onset/mel_cache`.

`official_ratio=0` now means **natural-frequency shuffle** (official stays in the pool). It used to zero official weights and silently drop them from train.

## Scoreboard (same 50-song `#2` val pack unless noted)

Judge **onset F1 + NPS band 1.2–2.5 + parity**. Token acc is padded by `BAR`/`POS`/`EMPTY`/`BOMB_EMPTY`.

| Run | overall TF | cube TF | gen F1 mean (med) | NPS | NPS in 1.2–2.5 | parity |
|-----|------------|---------|-------------------|-----|----------------|--------|
| `#2` customs Normal 0.90, 1 seed | 67.7% | 29.2% | 0.213 (0.223) | 1.75 | 35/50 | 0.218 |
| `#4` same + DLC @ 20% batch | 52.4% | **0.09%** | 0.079 (0.000) | 1.01 | — | 0.410 |
| DLC in pool, `official_ratio=0` | 64.8% | 16.1% | — | — | — | — |
| Cubes-only (`include_bombs: false`) | 57.5%* | 26.0% | 0.030 (0.009) | 0.17 | 0/50 sparse 46 | — |
| 4k maps @ 0.85, `#2` recipe | 65.5% | 31.1%† | 0.051 (0.017) | 0.80 | sparse 19/50 | 0.260 |
| **SS fine-tune from `#2`, mix 0.25, 1 seed** | 67.2% | 26.2% | 0.231 (0.256) | 1.61 | 41/50 | 0.224 |
| **SS + `--candidates 4` (SHIP)** | — | — | **0.259 (0.256)** | **1.75** | **49/50** | **0.185** |
| SS + onset mel bin + cand 4 | — | — | 0.246 (0.262) | 1.74 | 48/50 | 0.167 |
| Cube head v1, unweighted BCE, bias 3.0 | — | — | **0.0000** | 1.30 | 50/50 gate | — |
| Cube head v2, pos_weight 20, bias 1.0, epoch_003 (recall 0.571) | — | — | 0.230 (—) | 1.68 | 50/50 gate | 0.176 |
| **Cube head v2, epoch_007 (recall 0.572)** | — | — | **0.275 (—)** | **1.74** | **50/50 gate** | 0.194 |
| Cube head v2, epoch_007, `cube_head_bias_scale=0` ablation | — | — | 0.233 (—) | 1.76 | 50/50 gate | 0.170 |
| Cube head + **real DLC** (`processed_normal_with_dlc`), epoch_003 | — | — | 0.245 (—) | 1.85 | 50/50 gate | 0.185 |
| Cube head + **real DLC**, epoch_008 (final) | — | — | 0.251 (—) | 1.68 | 50/50 gate | 0.183 |
| Cube head v2, `--resume` ep7→ep15 (customs) | — | — | 0.243 (—) | 1.51 | 50/50 gate | 0.209 |
| Cube head v2, `--init-from` ep7, SS mix 0.4, epoch_001 / epoch_006 | — | — | 0.253 / 0.242 | 1.72 / 1.45 | 50/50 gate | 0.172 / 0.216 |
| Cube head v2 **seed 2**, epoch_005 (recall 0.830) | — | — | 0.254 (—) | 1.77 | 50/50 gate | 0.185 |
| Cube head v2 **seed 2**, epoch_006 (final, recall 0.675) | — | — | 0.250 (—) | 1.76 | 50/50 gate | 0.185 |

\*Not comparable to 67% — no `BOMB_EMPTY` freebies.  
†Partly leaky: adding songs reshuffles the 80/10/10 hash split. Own-val cube still ~31% vs `#2` 29%. Generate still died.

Class-wise TF (why 67% lies), `#2` val: cube 29%, cube_empty 85%, bomb **1%**, bomb_empty 100%, structure 65%. Bombs were never learned on `#2`. 0.85 taught bombs to 29% TF and still generated empty maps.

**Correction:** cube head v1/v2 both trained on `data/processed_normal_customs_only`, which is 100% `beatsaver`-source rows (verified directly — `read_notes_parquet(...)['source'].value_counts()` shows zero `official`). `build_weighted_sampler()` returns `None` whenever one source is entirely absent from the dataset (`dataset.py:558`), so the `official_ratio: 0.2` field left in `small_bombs_normal_cube_v2.json` was a no-op — **no DLC was actually used in the winning run**. This is a clean customs-only result, directly comparable to SS with no data-source confound. (An earlier version of this doc claimed the opposite — that was wrong, caught before spending a GPU run re-verifying it.) The bias-scale=0 ablation (0.233, below SS's 0.259) still shows the win isn't just a richer encoder from the auxiliary loss — the inference-time logit bias itself is what beats baseline (0.275 vs 0.233 on the same weights).

**Real-DLC follow-up:** same config, pointed at `data/processed_normal_with_dlc` (customs + 338 official) instead. This time the sampler was confirmed live — training log: `Weighted sampler: 270 official (w=1.4513), 1636 custom (mean_score=0.9580), target official_ratio=20%`. Result: F1 0.245–0.251, crushing `#4`'s 0.079 but ~0.025 short of the customs-only cube head (0.275) and just short of SS (0.259). Verdict: the cube head mostly fixes the DLC skip-onset collapse (it's no longer catastrophic), but DLC still mildly dilutes generate quality relative to customs-only — it does not make DLC "free." Ship the customs-only cube checkpoint (epoch_007), not this one.

Eval JSONs: `output/eval_ss_vs_2.json`, `output/eval_ss_candidates4.json`, `output/eval_ss_onset_candidates4.json`, `output/eval_cube_diag_*.json`, `output/eval_cube_v2_epoch3.json`, `output/eval_cube_v2_epoch7.json`, `output/eval_cube_v2_epoch7_nobias.json`, `output/eval_cube_dlc_epoch3.json`, `output/eval_cube_dlc_epoch8.json`, `output/eval_cube_v2_continue_epoch15.json`, `output/eval_cube_v2_ss2_epoch1.json`, `output/eval_cube_v2_ss2_epoch6.json`, `output/eval_cube_v2_seed2_epoch5.json`, `output/eval_cube_v2_seed2_epoch6.json`.

## Data on this machine

| Path | What |
|------|------|
| `data/raw/beatsaver_normal90/` | 4,000 BeatSaver maps (2,016 @ ≥0.90 + 1,984 @ 0.85–0.90) |
| `data/processed_normal_customs_only/` | **2,000 Normal-only**, 0.90. `#2` / SS train set. 80-bin mel cache. |
| `data/processed_normal_customs_085/` | 3,963 Normal-only from the 4k dump. Do not train generate on this. |
| `data/processed_normal_customs_onset/` | Copy of the 0.90 parquet + **81-bin** mel cache. |
| `data/processed_normal_with_dlc/` | Normal customs + 338 official. `#4` poison set. |
| `output/training_normal_customs_only/` | `#2` |
| `output/training_normal_ss_from_2/` | **SHIP** |
| `output/training_normal_ss_onset/` | Onset-bin fine-tune (do not ship) |
| `output/training_normal_with_dlc/` | `#4` collapsed |
| `output/training_normal_with_dlc_unweighted/` | DLC @ natural frequency; cubes 16% |
| `output/training_normal_cube_v2/` | Cube head, customs-only, seed 1. `checkpoints/epoch_007` = best single number on the board (F1 0.275). Not reproduced by seed 2. |
| `output/training_normal_cube_v2_seed2/` | Same command, seed 2. F1 0.250–0.254 (ep5/ep6). Cube head ≈ SS within noise. |
| `output/training_normal_cube_dlc/` | Cube head, real DLC mix. F1 0.245–0.251 — better than `#4`, still short of cube_v2. |
| `output/training_normal_cube_v2_continue/` | Cube head continued past epoch_007 (ep8–15, customs-only). F1 fell to 0.243 — do not use, confirms epoch_007 is the peak. |
| `output/training_normal_cube_v2_ss2/` | Cube head, `--init-from epoch_007`, SS mix 0.25→0.4. F1 0.242–0.253 — also below 0.275, do not use. |

## Code from this session (uncommitted unless noted)

Branch `feature/bombs-walls-official-data` is also **10 local commits ahead of** `fork/feature/bombs-walls-official-data` (arcs pipeline, windowed-gen fixes, difficulty filters, official-use note). Do not dump those onto [PR #17](https://github.com/asfilion/beat-weaver/pull/17) as one PR.

This session added:

- `evaluate --mode teacher-forced|generate|both`, `--split`, `--max-maps`, `--candidates`, class-wise TF acc (cube / cube_empty / bomb / structure).
- Generate playability **gate** (NPS floor + parity ≤ 0.40 + min notes) and **`--candidates`** picker (`select_playable_candidate`).
- Parallel **scheduled sampling** (`scheduled_sampling_prob`, `scheduled_sampling_ramp_epochs`).
- `train --init-from` (weights only). `--resume` + `--init-from` together is an error.
- Onset-channel **weight expand**: 80→81 `input_proj`, new column zero-init.
- `official_ratio <= 0` → sampler `None` (natural frequency), not official weight 0.
- Binary **cube head**: `use_cube_head`, `cube_head_weight`, `cube_head_bias_scale`, `cube_head_pos_weight` config fields; `cube_target_from_tokens`/`cube_head_loss` in `training.py`; inference-time POS-logit bias in `inference.py::generate()`; `val_cube_recall` diagnostic (accuracy alone hides the majority-class collapse).
- Configs: `small_bombs_normal_ss.json`, `small_bombs_normal_ss_onset.json`, `small_bombs_normal_dlc_unweighted.json`, `small_normal_only.json`, `small_bombs_normal_cube.json` (v1, do not use — unweighted BCE), `small_bombs_normal_cube_v2.json` (pos_weight fix, current best — `--data data/processed_normal_customs_only` for the ship result, `--data data/processed_normal_with_dlc` for the DLC retest), `small_bombs_normal_cube_v2_continue.json` (same recipe, `max_epochs`/`early_stopping_patience` raised for the epoch_007→015 continuation — do not use, F1 regressed), `small_bombs_normal_cube_v2_ss2.json` (`--init-from epoch_007`, SS mix 0.25→0.4 — do not use, F1 also regressed).

## Reproduce the ship eval

```bash
beat-weaver evaluate \
  --checkpoint output/training_normal_ss_from_2/checkpoints/best \
  --data data/processed_normal_customs_only \
  --audio-manifest data/audio_manifest_normal_customs_only.json \
  --split val --mode generate --max-maps 50 --candidates 4 \
  --output output/eval_ss_candidates4.json
```

Fine-tune recipe that produced SS (do not `--resume` `#2`):

```bash
beat-weaver train \
  --config configs/small_bombs_normal_ss.json \
  --data data/processed_normal_customs_only \
  --audio-manifest data/audio_manifest_normal_customs_only.json \
  --output output/training_normal_ss_from_2 \
  --init-from output/training_normal_customs_only/checkpoints/best
```

## Next experiment (when you want training)

**Binary "cube here" head: done. Beat SS on one seed (0.275); a second seed landed at 0.250–0.254, below SS. Net: parity with SS within seed noise, not a confirmed win — see the seed-2 note below.** `configs/small_bombs_normal_cube_v2.json` adds `cube_head` (per-frame binary event head on the encoder, `cube_head_pos_weight=20.0` in the BCE loss, `cube_head_bias_scale=1.0` added to POS logits at inference). Epoch_007 (`output/training_normal_cube_v2/checkpoints/epoch_007`) got **0.275 generate F1**, beating SS's 0.259. See Scoreboard. `load_weights()` now tolerates a missing `cube_head.*` when loading an SS/`#2` checkpoint into a cube-head model (`strict=False` + explicit expected-missing-keys check), so `--init-from` across the head-shape change works.

Caveats before shipping this over SS:

- This run trained on `data/processed_normal_customs_only` — same customs-only, 100%-BeatSaver data as SS/`#2`, no DLC (its `official_ratio: 0.2` field was a leftover no-op, see Scoreboard correction). So this checkpoint is already shareable-equivalent to SS on the data-source question — no separate DLC/no-DLC comparison needed for that.
- Only one seed per point on this table, same as everything else in it — no variance estimate yet.
- `val_loss`-selected "best" checkpoint (epoch_002) has `cube_recall≈0.06` (bad) — always pick by `val_cube_recall` + generate F1 for this head, not val_loss.

**Real DLC retest (customs + 338 official, `data/processed_normal_with_dlc`):** sampler confirmed live this time (`Weighted sampler: 270 official (w=1.4513), 1636 custom, target official_ratio=20%`, not a no-op). Result: F1 0.245 (epoch_003) / 0.251 (epoch_008) — crushes `#4`'s 0.079, but ~0.025 short of the customs-only cube head. **Verdict: cube head mostly fixes the DLC skip-onset collapse, but doesn't make DLC beat customs-only.** Ship the customs-only checkpoint.

**Continuing past epoch_007 (customs-only, epochs 8–15, same recipe, `--resume`, `early_stopping_patience` raised to 10 so it wouldn't cut off on stale val_loss):** generate F1 **fell** to 0.243 by epoch_015, NPS softened to 1.51, parity got slightly worse (0.209). Train loss kept dropping through all 9 extra epochs while val_loss stayed flat/rose slightly — classic overfitting-to-teacher-forcing, exactly the same exposure-bias story as every other "more training" attempt in this project.

**Higher scheduled-sampling mix from epoch_007** (`small_bombs_normal_cube_v2_ss2.json`, `--init-from epoch_007` — fresh optimizer/LR, not `--resume` — `scheduled_sampling_prob` 0.25→0.4, ramped over 2 epochs, 6-epoch run): also regressed. epoch_001 (mix still ramping, ~0.2) scored 0.253; epoch_006 (full 0.4 mix) scored 0.242 — both below 0.275, and the further the mix was pushed, the worse it got. Same conclusion from a second, independent direction.

**`epoch_007` is a genuine local peak for seed 1's trajectory, not an artifact of stopping early.** Two independent follow-ups — more plain epochs, and a higher scheduled-sampling mix — both degrade generate F1 roughly in proportion to how far they push past it. **Do not keep training this checkpoint further.**

**Seed 2 (identical command, `output/training_normal_cube_v2_seed2`; training has no fixed seed, only the split does):** early-stopped at epoch 6. The head over-fired mid-run (epoch_004 recall 0.91 / acc 0.23) before settling. epoch_005 → **0.254**, epoch_006 → **0.250**. Both below seed 1's 0.275 and below SS's 0.259. Seed-to-seed spread on the cube head (~0.02–0.025) is larger than the cube-vs-SS gap (0.016).

**Revised verdict: the cube head is at parity with SS on generate F1 within seed noise, not a confirmed win.** Two cube seeds average ~0.263 vs SS's one-seed 0.259. What *is* consistent across both cube seeds: gate 50/50 (SS: 49/50), NPS 1.74–1.77 squarely in band, and the DLC retest showing the head prevents the `#4` collapse. So it's a robustness gain, not an F1 gain. SS is also one seed — a fair head-to-head needs an SS seed 2, but that only sharpens a tie; it won't turn this into a clear win. `output/training_normal_cube_v2/checkpoints/epoch_007` remains the best single number on the board; do not read that as "better than SS." Next gains on this line need a different lever (candidate/gate tuning, section-density budgeting, held-out-mapper eval), not more epochs, more scheduled sampling, or more seeds.

Still true from the original cut:

- Keep `#2` 0.90 Normal data as the base, bombs on, `--candidates 4` at eval, early-stop / pick on **generate F1 + NPS band**, not val_loss.

Player-sim as **critic** (parity / empty-NPS gate) is already in generate. Do not train the mapper on "what a bot likes to slash."

## Traps

- Hash split is over all songs in a processed dir. Adding maps **moves** who is in val. `#2` val is not a clean holdout for the 0.85 model (156/200 leaked into 0.85 train).
- `evaluate` generate uses `generate()` (single window). Production `generate` CLI uses `generate_full_song()`. F1 numbers are comparable across eval runs, not identical to a shipped long song.
- Mel cache `VERSION` is per processed dir. Training with `use_onset_features` on the 0.90 dir would **delete** the 80-bin cache. That is why onset used a copy dir.
- TensorBoard events are under `output/<run>/logs/`, not `tensorboard/`.
