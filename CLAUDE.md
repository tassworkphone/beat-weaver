# Beat Saber Track Generator

An AI-powered mod for Beat Saber that generates custom note maps from audio files.

## Project Overview

Given an audio file as input, the system generates block positions and orientations (for both left and right sabers) at a regular sample rate (~20 Hz). The goal is a machine learning model that produces playable, enjoyable Beat Saber tracks automatically.

## Architecture

1. **Data Pipeline** (complete) — Extract, parse, normalize Beat Saber maps into Parquet
2. **ML Model** (complete) — Encoder-decoder transformer with Conformer audio encoder for audio → token sequence generation
3. **Baseline Training** (complete) — 16 epochs on 23K songs, 60.6% token accuracy, generates playable maps
4. **Model Improvements** (complete) — Filtering, SpecAugment, onset features, RoPE, color balance loss, medium config
5. **Conformer Encoder** (complete) — Conformer blocks (conv + attention) replace pure transformer encoder, now default
6. **Full-Song Generation** (complete) — Windowed inference with overlap stitching for songs of any length
7. **Bombs** (complete) — Model-native compound token (`include_bombs` config flag, opt-in, vocab 291→304)
8. **Walls/Obstacles** (complete) — Rule-based post-processing (`beat_weaver.model.obstacles`), stats mined from real Parquet data with outlier filtering (community "wall art" maps pollute raw obstacle data — see Open Questions)
9. **Official DLC as training data** (complete) — `extract-official` pulls base game + all owned DLC (332 levels extracted from a full-DLC install, far more than the original 214); official maps get oversampled via `official_ratio` for quality/consistency
10. **Arcs** (not started, planned as rule-based post-processing like walls — see Open Questions) / **Chains** (not started, no validated approach yet)
11. **Feedback System / Self-Improvement Loop** (future) — See Open Questions

## Beat Saber Map Format Quick Reference

Maps are **JSON files** in folders. See [RESEARCH.md](RESEARCH.md) for full format details. Implementation plans are in [plans/](plans/).

- **Schemas:** v2 (most custom maps), v3, v4 (latest official). All JSON-based, `.dat` extension.
- **Entry point:** `Info.dat` — song metadata, references difficulty files
- **Grid:** 4 columns (x: 0-3) x 3 rows (y: 0-2)
- **Colors:** 0 = Red/Left, 1 = Blue/Right
- **Cut directions:** 0=Up, 1=Down, 2=Left, 3=Right, 4=UpLeft, 5=UpRight, 6=DownLeft, 7=DownRight, 8=Any
- **Timing:** All in beats (float). Convert: `seconds = beat * 60.0 / BPM`
- **Custom maps location:** `<Beat Saber>/Beat Saber_Data/CustomLevels/`
- **Official maps:** Extracted from Unity bundles via `UnityPy` — 214 levels (65 base + 149 DLC) with WAV audio
- **DLC maps location:** `<Beat Saber>/DLC/Levels/<LevelName>/<bundlefile>` — individual bundles per level
- **Training data sources:** Custom maps from [BeatSaver](https://beatsaver.com/) (v2 JSON, score>=0.9/upvotes>=5) + 214 official levels (v4 gzip JSON + WAV audio)
- **Local install:** `C:\Program Files (x86)\Steam\steamapps\common\Beat Saber`

## CLI (`beat-weaver`)

Install: `pip install -e .` (core) or `pip install -e ".[ml]"` (with ML dependencies)

| Command | Description |
|---------|-------------|
| `beat-weaver download` | Download custom maps from BeatSaver API |
| `beat-weaver extract-official` | Extract official maps from Unity bundles |
| `beat-weaver build-manifest` | Build audio manifest from raw map folders |
| `beat-weaver process` | Normalize raw maps to Parquet |
| `beat-weaver run` | Full pipeline (all sources) |
| `beat-weaver train` | Train the ML model |
| `beat-weaver generate` | Generate a Beat Saber map from audio (bombs if checkpoint trained with them; walls always, unless `--no-obstacles`) |
| `beat-weaver analyze-obstacles` | Mine real wall placement stats (density/duration/width/crouch-ratio) from processed data, filtering out modded/"wall art" outliers |
| `beat-weaver evaluate` | Evaluate model on test data |

**Key modules:**
- `beat_weaver.parsers.beatmap_parser.parse_map_folder(path)` — parse any map folder
- `beat_weaver.sources.beatsaver` — BeatSaver API client + downloader
- `beat_weaver.sources.unity_extractor` — official map + audio extraction from Unity bundles (base + DLC)
- `beat_weaver.storage.writer` — Parquet output (notes/bombs/obstacles)
- `beat_weaver.model.tokenizer` — encode/decode beatmaps ↔ token sequences (291 vocab, or 304 with `include_bombs=True`)
- `beat_weaver.model.obstacles` — rule-based wall/obstacle placement (mine stats from Parquet, generate walls in gaps between generated notes)
- `beat_weaver.model.audio` — mel spectrogram extraction, beat-aligned framing, BPM auto-detection
- `beat_weaver.model.transformer` — AudioEncoder (Conformer or Transformer) + TokenDecoder + BeatWeaverModel (RoPE or sinusoidal PE)
- `beat_weaver.model.inference` — autoregressive generation with grammar mask + windowed full-song generation
- `beat_weaver.model.exporter` — token sequence or note list → playable v2 map folder
- `beat_weaver.model.evaluate` — onset F1, NPS accuracy, parity, diversity metrics

**Output format:** `data/processed/notes_NNNN.parquet` (one row group per song, split at 1 GB) with columns: song_hash, source, difficulty, characteristic, bpm, beat, time_seconds, x, y, color, cut_direction, angle_offset. Reader (`read_notes_parquet`) handles both multi-file and legacy single-file layouts.

**Model configs:** `configs/small.json` (1M params, batch_size=32) for fast iteration. `configs/small_bombs.json` (same size, `include_bombs=true`, vocab_size=304) — the CPU/small-GPU-appropriate config with bombs. `configs/medium.json` (6.5M params, 4L/256d) for standard transformer. `configs/medium_conformer.json` (9.4M params, Conformer, 8GB VRAM). `configs/large_conformer.json` (62M params, 6L/512d, Conformer, 24GB+ VRAM) for full training.

**Tests:** `python -m pytest tests/ -v` (178 tests; ML tests skipped without `.[ml]` deps)

**Training data:** 23,588 songs (23,375 BeatSaver + 213 official), 42,542 training samples, 40.3M notes total. Mel spectrograms pre-cached to `data/processed/mel_cache/` (~23K `.npy` files, ~30GB). Cache auto-invalidates when audio feature config changes (VERSION file).

## ML Model

See [RESEARCH.md](RESEARCH.md) for research details, [plans/002-ml-model.md](plans/002-ml-model.md) for implementation plan.

- **Architecture:** Encoder-decoder transformer (62M large conformer, 9.4M medium conformer, 6.5M medium standard, 1M small)
- **Audio encoder:** Conformer (default) or standard Transformer. Conformer blocks use FFN/2 + Self-Attention + DepthwiseConv + FFN/2 + LayerNorm (Gulati et al., 2020). Config: `use_conformer=True` (default), `conformer_kernel_size=31`.
- **Audio input:** Log-mel spectrogram (80 bins, sr=22050, hop=512), beat-aligned to 1/16th note grid. Optional onset strength channel (+1 bin).
- **Positional encoding:** RoPE (default) or sinusoidal (config.use_rope=False). RoPE applied to self-attention Q/K only (not cross-attention).
- **Scope:** Color notes (model-native) + bombs (model-native, opt-in `include_bombs`) + walls (rule-based post-processing, not model-native — see `beat_weaver.model.obstacles`). Arcs and chains not yet implemented (no parser support) — see Open Questions; arcs are planned as rule-based post-processing too (same shape as walls), chains have no validated approach yet.
- **Output:** Beat-quantized compound tokens (291 vocab) → v2 Beat Saber JSON
- **Token format:** `START DIFF BAR POS LEFT RIGHT ... BAR ... END`
- **Training:** Cross-entropy with label smoothing + optional color balance auxiliary loss, AdamW + cosine LR, mixed-precision, early stopping, weighted sampling (official 20% of batch, custom weighted by BeatSaver score)
- **Data augmentation:** SpecAugment (time/frequency masking, training split only)
- **Data filtering:** By difficulty (min_difficulty), characteristic, BPM range. Out-of-grid notes filtered during pre-tokenization.
- **Inference:** Autoregressive with grammar-constrained decoding (strictly increasing POS per bar), temperature/top-k/top-p sampling, windowed full-song generation with overlap stitching
- **Evaluation:** Onset F1, parity violation rate, NPS accuracy, beat alignment, pattern diversity
- **Mel pre-caching:** `warm_mel_cache()` computes all spectrograms in parallel before training starts (ProcessPoolExecutor, ~25min for 14K songs). Cache versioned and auto-invalidated on config change.
- **Full-song generation:** `generate_full_song()` processes audio in overlapping windows (25% overlap, capped at 1024 frames), decodes each window's tokens to notes with beat offsets, and merges using midpoint ownership to eliminate duplicates. Songs of any length are supported. `export_notes()` writes a merged note list directly to v2 map format.

## Training Results

### Baseline (small config, 1M params, 23K songs)

Best model after 16 epochs: **val_loss=2.055, 60.6% token accuracy**. Model plateaued — 1M params saturated on 42K samples. Generates playable maps in-game. Known issues: color imbalance (skews red), NPS sometimes too high/low.

### Medium Standard Transformer (6.5M params)

Training diverged after epoch 4 with `learning_rate=1e-4`: train loss dropped (4.26 → 2.0) but val loss exploded (5.26 → 13.4). Root cause: LR too aggressive for the larger model — not overfitting but optimization instability.

### Medium Conformer (9.4M params, Expert+ only) — current best

Training with `configs/medium_conformer.json` (LR=3e-5, warmup=1000, batch_size=8). Best model at epoch 26/50: **val_loss=2.2324, 59.44% token accuracy**. Training continued 10 more epochs with no val loss improvement (early stopping). Train loss continued decreasing, indicating the model began overfitting on the Expert+-only dataset (~18K samples).

| Epoch | Train Loss | Val Loss | Val Acc |
|-------|-----------|----------|---------|
| 1 | 4.046 | 3.277 | 37.8% |
| 7 | 2.189 | 2.396 | 53.6% |
| **26** | **—** | **2.232** | **59.4%** |
| 36 | — | plateau | early stop |

Comparable to the small model baseline (60.6% on all difficulties) despite training on Expert+ only (~18K samples vs 42K). The larger model with less data hit its ceiling — next step is broadening the data filter.

**Key lesson:** Medium+ models need lower LR (3e-5 vs 1e-4) and longer warmup (1000 vs 500 steps). The Conformer's conv module helps capture local audio patterns (onsets, transients) that pure attention misses.

**Checkpoint resume:** Must save ALL training state (model, optimizer, scheduler, GradScaler). Missing GradScaler was root cause of NaN on resume (fresh scaler at scale=65536 causes overflow). Now saves `scheduler.pt` + `scaler.pt`; fallbacks handle old checkpoints.

### Small + Bombs (1.4M params, GPU)

`configs/small_bombs.json`, 30 epochs, ~1,963 BeatSaver songs (all difficulties): **val_loss=1.9663, 67.5% token accuracy** (epoch 27) — notably above the original 60.6% baseline, on GPU in ~68 min (RTX 3060). A second run added 332 extracted official DLC levels (~2,295 songs total, oversampled via `official_ratio`) — see `output/training_v2/` once complete.

**Note on the ~60-68% accuracy ceiling:** likely dominated by the one-to-many nature of mapping (many valid note patterns exist for the same audio; token accuracy penalizes any deviation from one specific reference mapper's choice) plus mixed mapper skill/style in the BeatSaver corpus — not primarily a model-capacity problem. Evidence: scaling 1M→9.4M params barely moved accuracy (60.6%→59.4%). The onset F1 / parity violation / NPS accuracy / pattern diversity metrics in `evaluate.py` are more meaningful quality signals than raw token accuracy for this reason.

## Open Questions

- **Data filter broadening:** Expert+ only (~18K samples) limits the 9.4M param Conformer. Adding Expert maps would roughly double the dataset. (Moot for `small*.json` configs, which default to `min_difficulty="Easy"` — already maximally broad.)

- **Arcs — newly unlocked by official DLC data, and simpler than first thought.** No parser support exists yet (`schemas/v2.py`/`v3.py`/`v4.py` don't read `arcs`/`arcsData` fields at all), but the extracted official DLC `.dat` files (`data/raw/official/*/*.dat`) already contain them natively in v4 JSON — most official tracks have 50-170+ arc events each, cleanly placed. **Revised plan (originally guessed model-native, like bombs — corrected after reading a more advanced prior-art project's actual source, [InfernoSaber's `map_creation/gen_sliders.py`](https://github.com/fred-brenner/InfernoSaber---BeatSaber-Automapper)):** arcs don't need to touch the model or tokenizer at all. InfernoSaber generates arcs as pure rule-based post-processing exactly like our wall module: after notes are generated, walk consecutive same-color notes and connect eligible pairs with an arc (head/tail = the two notes' own positions/directions). Their tuned reference constants: `slider_time_gap=[0.5, 12.0]s` (only connect notes in this timing window), `slider_movement_minimum=3` (skip near-stationary pairs), `slider_probability=0.8` (randomly skip ~20% of eligible pairs so it's not wall-to-wall arcs) — worth mining our own equivalents from real official-map arc data the same way `analyze-obstacles` mines wall stats, rather than hand-picking constants. Implementation mirrors `beat_weaver.model.obstacles`: (1) `Arc` dataclass in `schemas/normalized.py`, (2) read `arcs`/`arcsData` in `v4.py`, (3) Parquet schema/writer/reader (mirroring `bombs`/`obstacles`), (4) `mine_arc_stats()` + `generate_arcs(notes, ...)` analogous to `mine_obstacle_stats()`/`generate_obstacles()`, consumed the same way in `cmd_generate`/`export_notes()`. Re-running `process` (not `extract-official`) is sufficient once parser support lands — no new extraction needed.

- **Chains — still the harder piece, no *validated* shortcut found yet.** Same missing-parser situation as arcs (`chains`/`chainsData` fields exist in the extracted official v4 JSON, unread). InfernoSaber has no chain-generation file at all — even that more automation-complete prior-art project treats chains as out of scope.

  A second project, [Kwoolford/beatsaber_automapper](https://github.com/Kwoolford/beatsaber_automapper), has a well-designed *candidate* model-native token encoding worth keeping as a reference if we ever attempt this: `CHAIN_HEAD = [HAND] [Δt] [CHAIN_HEAD] [X] [Y] [DIR] [SLICE_bin]` (slice count 2-32, quantized into 31 bins) and `CHAIN_TAIL = [HAND] [Δt] [CHAIN_TAIL] [TAIL_X] [TAIL_Y] [SQUISH_bin]` (squish factor 0.0-1.0, 11 bins), FIFO-paired per hand — the same head/tail-pairing shape we already use for bombs, just with two extra binned fields. **Caveat, checked directly: this is unvalidated.** That project's own `PROGRESS.md` shows it has since abandoned the transformer/token-grammar approach entirely in favor of a non-ML rule-based "builder" system, with unresolved structural issues in its own status notes and zero reported training metrics for the grammar — the token spec was apparently never actually trained to convergence. Combined with InfernoSaber's total lack of chain support, that's two independent signals that model-native generation of structurally complex elements (chains specifically — a variable-length segment count plus a squish curve, not just a point placement like a bomb) is harder in practice than it looks on paper. Treat the token encoding above as a starting design if we do attempt model-native chains, not as a proven path — budget for it to need real iteration, and don't assume it'll work as cleanly as bombs did on the first try.

  Lowest priority of the three remaining map elements; revisit after arcs (which do have a validated rule-based path) are working and generating real value.

- **Self-improvement loop (generate → filter → retrain) — plan, not yet built.** The idea: once a decent checkpoint (with bombs/walls/arcs/chains) exists, generate a batch of new maps with it, and feed the *good* ones back into training to bootstrap a better model. **This must not be done naively** — feeding a model's own unfiltered outputs back into its own training data is a well-documented failure mode ("model collapse": the model only reinforces patterns it already has, including its existing flaws like color imbalance, and quality drifts down over generations rather than up, since no new information is introduced). The loop only works with a **quality filter** between generation and retraining:
  1. Generate a batch of candidate maps from the current best checkpoint (varying seed/temperature/source songs).
  2. **Filter** — keep only maps that pass a quality bar, via either:
     - *Automated*: score with `beat_weaver.model.evaluate`'s existing metrics (onset F1, parity violation rate, NPS accuracy, pattern diversity) and threshold.
     - *Human*: actual player feedback (fun/playable rating) — this is the original "Feedback capture system" / "RL fine-tuning" idea below, generalized.
  3. **Mix, don't replace** — add the filtered generated maps into training *alongside* the full real dataset (BeatSaver + official), at a modest ratio (similar spirit to `official_ratio`'s existing oversampling knob), never as a replacement for real data.
  4. Retrain/fine-tune from the current checkpoint (low LR, short run — see fine-tuning caveats: a `--resume` reuses the original run's fully-decayed cosine LR schedule, so a dedicated low-LR short-schedule config is needed for this, not the original training config as-is).
  - Prerequisite: this is worth building only once chains/arcs and general map quality are solid — bootstrapping from a weak model mostly just amplifies its weaknesses faster.

- **Feedback capture system:** In-game mechanism to collect player feedback (later phase) — the human-filter half of the self-improvement loop above.
- **RL fine-tuning:** After supervised pretraining, fine-tune with a player-feedback reward model — a further formalization of the self-improvement loop using real human ratings as the reward signal instead of (or alongside) automated heuristic metrics.
