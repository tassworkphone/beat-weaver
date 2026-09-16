"""Command-line interface for Beat Weaver data pipeline."""

import argparse
import logging
from pathlib import Path


def cmd_download(args: argparse.Namespace) -> None:
    from beat_weaver.sources.beatsaver import BeatSaverClient

    client = BeatSaverClient()

    # Count already-downloaded maps for the summary
    dest = Path(args.output)
    existing = sum(1 for d in dest.iterdir() if d.is_dir()) if dest.exists() else 0

    downloaded = client.download_maps(
        dest_dir=dest,
        min_score=args.min_score,
        min_upvotes=args.min_upvotes,
        max_maps=args.max_maps,
        workers=args.workers,
        difficulties=args.difficulty,
    )
    newly = len(downloaded) - existing if len(downloaded) > existing else len(downloaded)
    limit = f" (limit: {args.max_maps})" if args.max_maps > 0 else " (no limit)"
    print(f"Done: {len(downloaded)} maps in {args.output} ({newly} new){limit}")


def cmd_extract_official(args: argparse.Namespace) -> None:
    from beat_weaver.sources.unity_extractor import extract_official_maps

    bundles_dir = (
        Path(args.beat_saber)
        / "Beat Saber_Data"
        / "StreamingAssets"
        / "aa"
        / "StandaloneWindows64"
    )
    extracted = extract_official_maps(
        bundles_dir, Path(args.output), beat_saber_path=Path(args.beat_saber),
    )
    print(f"Extracted {len(extracted)} map folders to {args.output}")


def cmd_build_manifest(args: argparse.Namespace) -> None:
    from beat_weaver.model.audio import build_audio_manifest, save_manifest

    raw_dirs = [Path(d) for d in args.input]
    manifest = build_audio_manifest(raw_dirs)
    save_manifest(manifest, Path(args.output))
    print(f"Built audio manifest: {len(manifest)} entries -> {args.output}")


def _detect_source(map_folder: Path, input_root: Path) -> str:
    """Detect map source from its path relative to the input root."""
    try:
        rel = map_folder.relative_to(input_root)
        parts = [p.lower() for p in rel.parts]
    except ValueError:
        parts = []
    if "official" in parts:
        return "official"
    if "beatsaver" in parts:
        return "beatsaver"
    return "local_custom"


def _process_single_folder(
    map_folder: Path, source: str,
) -> list:
    """Process a single map folder (top-level for pickling by ProcessPoolExecutor)."""
    from beat_weaver.pipeline.processor import process_map_folder
    from beat_weaver.sources.beatsaver import load_beatsaver_meta

    beatmaps = process_map_folder(map_folder, source=source, source_id=map_folder.name)

    # Inject BeatSaver score/votes into metadata if available
    if source == "beatsaver":
        meta = load_beatsaver_meta(map_folder)
        if meta:
            stats = meta.get("stats", {})
            score = stats.get("score")
            upvotes = stats.get("upvotes")
            downvotes = stats.get("downvotes")
            for bm in beatmaps:
                if score is not None:
                    bm.metadata.score = score
                if upvotes is not None:
                    bm.metadata.upvotes = upvotes
                if downvotes is not None:
                    bm.metadata.downvotes = downvotes

    return beatmaps


def _filter_by_difficulty(beatmaps: list, wanted: set[str] | None) -> tuple[list, int]:
    """Keep only beatmaps whose difficulty is in *wanted*.

    Returns (kept, num_excluded). wanted=None means no filtering — every
    beatmap is kept, matching the pre-filter default behavior exactly.
    """
    if wanted is None:
        return beatmaps, 0
    kept = [bm for bm in beatmaps if bm.difficulty_info.difficulty in wanted]
    return kept, len(beatmaps) - len(kept)


def cmd_process(args: argparse.Namespace) -> None:
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from beat_weaver.storage.writer import write_parquet

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    # Collect all map folders up front
    folders = []
    for info_file in sorted(input_dir.rglob("Info.dat")):
        map_folder = info_file.parent
        source = _detect_source(map_folder, input_dir)
        folders.append((map_folder, source))

    wanted_difficulties = set(args.difficulty) if getattr(args, "difficulty", None) else None

    all_beatmaps = []
    skipped_difficulty = 0
    with ProcessPoolExecutor() as executor:
        futures = {
            executor.submit(_process_single_folder, folder, source): folder
            for folder, source in folders
        }
        for future in as_completed(futures):
            try:
                result, excluded = _filter_by_difficulty(future.result(), wanted_difficulties)
                skipped_difficulty += excluded
                all_beatmaps.extend(result)
            except Exception:
                logging.getLogger(__name__).warning(
                    "Failed to process %s", futures[future], exc_info=True,
                )

    write_parquet(all_beatmaps, output_dir)
    if wanted_difficulties is not None:
        print(
            f"Processed {len(all_beatmaps)} beatmaps to {output_dir} "
            f"(excluded {skipped_difficulty} non-{'/'.join(sorted(wanted_difficulties))} difficulties)"
        )
    else:
        print(f"Processed {len(all_beatmaps)} beatmaps to {output_dir}")


def cmd_train(args: argparse.Namespace) -> None:
    from beat_weaver.model.audio import build_audio_manifest, save_manifest, load_manifest
    from beat_weaver.model.config import ModelConfig
    from beat_weaver.model.dataset import BeatSaberDataset, warm_mel_cache
    from beat_weaver.model.training import train

    config = ModelConfig()
    if args.config:
        config = ModelConfig.load(Path(args.config))
    if args.epochs:
        config.max_epochs = args.epochs
    if args.batch_size:
        config.batch_size = args.batch_size

    manifest_path = Path(args.audio_manifest)
    if not manifest_path.exists():
        print(f"Audio manifest not found: {manifest_path}")
        return

    data_dir = Path(args.data)

    # Pre-compute mel spectrograms in parallel before training
    warm_mel_cache(data_dir, manifest_path, config)

    train_ds = BeatSaberDataset(data_dir, manifest_path, config, split="train")
    val_ds = BeatSaberDataset(data_dir, manifest_path, config, split="val")

    print(f"Training: {len(train_ds)} samples, Validation: {len(val_ds)} samples")

    resume = Path(args.resume) if args.resume else None
    best_ckpt = train(config, train_ds, val_ds, Path(args.output), resume_from=resume)
    print(f"Training complete. Best checkpoint: {best_ckpt}")


# Audio file extensions scanned when --audio-dir is given.
_AUDIO_EXTENSIONS = {".wav", ".ogg", ".egg", ".mp3", ".flac", ".m4a"}


def _generate_one(model, config, device, args, audio_path: Path, output: Path) -> None:
    import torch
    from beat_weaver.model.audio import (
        beat_align_spectrogram, compute_mel_spectrogram, compute_mel_with_onset,
        detect_bpm, load_audio,
    )
    from beat_weaver.model.exporter import export_multi_difficulty
    from beat_weaver.model.inference import generate_full_song
    from beat_weaver.model.obstacles import generate_obstacles, load_obstacle_stats

    audio, sr = load_audio(audio_path, sr=config.sample_rate)

    bpm = args.bpm
    if bpm is None:
        bpm = detect_bpm(audio, sr=sr)
        print(f"Auto-detected BPM: {bpm:.1f}")

    if config.use_onset_features:
        mel = compute_mel_with_onset(audio, sr=sr, n_mels=config.n_mels,
                                     n_fft=config.n_fft, hop_length=config.hop_length)
    else:
        mel = compute_mel_spectrogram(audio, sr=sr, n_mels=config.n_mels,
                                      n_fft=config.n_fft, hop_length=config.hop_length)

    # Beat-align to match training data preprocessing
    mel = beat_align_spectrogram(mel, sr=sr, hop_length=config.hop_length, bpm=bpm)
    mel_tensor = torch.from_numpy(mel)

    n_windows = max(1, (mel.shape[1] - 1) // (config.max_audio_len - min(config.max_audio_len // 4, 1024)) + 1) if mel.shape[1] > config.max_audio_len else 1

    obstacle_stats = load_obstacle_stats(Path(args.obstacle_stats)) if args.obstacle_stats else None

    maps = {}
    summary = []
    for difficulty in args.difficulty:
        all_events = generate_full_song(
            model, mel_tensor, difficulty, config, bpm,
            temperature=args.temperature,
            seed=args.seed,
        )
        # decode_tokens (called inside generate_full_song) represents bombs as
        # color=3 Note entries — split them back out before obstacle placement/export.
        notes = [n for n in all_events if n.color in (0, 1)]
        bombs = [n for n in all_events if n.color == 3]

        obstacles = []
        if not args.no_obstacles:
            obstacles = generate_obstacles(notes, bombs, bpm, difficulty, stats=obstacle_stats)

        maps[difficulty] = (notes + bombs, obstacles)
        summary.append(f"{difficulty}: {len(notes)} notes, {len(bombs)} bombs, {len(obstacles)} obstacles")

    song_name = audio_path.stem
    export_multi_difficulty(maps, bpm, song_name, audio_path, output)
    print(f"Generated map: {output} ({n_windows} window(s) per difficulty)")
    for line in summary:
        print(f"  {line}")


def cmd_generate(args: argparse.Namespace) -> None:
    import torch
    from beat_weaver.model.config import ModelConfig
    from beat_weaver.model.transformer import BeatWeaverModel

    ckpt_dir = Path(args.checkpoint)
    config = ModelConfig.load(ckpt_dir / "config.json")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BeatWeaverModel(config)
    model.load_state_dict(
        torch.load(ckpt_dir / "model.pt", map_location="cpu", weights_only=True),
    )
    model.to(device)
    model.eval()

    if args.audio_dir:
        audio_dir = Path(args.audio_dir)
        audio_paths = sorted(
            p for p in audio_dir.iterdir()
            if p.is_file() and p.suffix.lower() in _AUDIO_EXTENSIONS
        )
        if not audio_paths:
            print(f"No audio files found in {audio_dir}")
            return
        base_output = Path(args.output) if args.output else Path("output")
        print(f"Queued {len(audio_paths)} song(s) from {audio_dir}")

        failures = []
        for i, audio_path in enumerate(audio_paths, start=1):
            output = base_output / audio_path.stem
            print(f"\n[{i}/{len(audio_paths)}] {audio_path.name}")
            try:
                _generate_one(model, config, device, args, audio_path, output)
            except Exception as exc:
                logging.getLogger(__name__).exception("Failed to generate map for %s", audio_path)
                failures.append((audio_path.name, str(exc)))

        print(f"\nDone: {len(audio_paths) - len(failures)}/{len(audio_paths)} succeeded")
        for name, err in failures:
            print(f"  FAILED {name}: {err}")
    else:
        song_name = Path(args.audio).stem
        output = Path(args.output) if args.output else Path(f"output/{song_name}")
        _generate_one(model, config, device, args, Path(args.audio), output)


def cmd_evaluate(args: argparse.Namespace) -> None:
    import json as _json
    import torch
    from beat_weaver.model.audio import (
        compute_mel_spectrogram, load_audio, load_manifest, beat_align_spectrogram,
    )
    from beat_weaver.model.config import ModelConfig
    from beat_weaver.model.dataset import BeatSaberDataset
    from beat_weaver.model.evaluate import evaluate_map
    from beat_weaver.model.inference import generate
    from beat_weaver.model.tokenizer import decode_tokens
    from beat_weaver.model.transformer import BeatWeaverModel

    ckpt_dir = Path(args.checkpoint)
    config = ModelConfig.load(ckpt_dir / "config.json")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BeatWeaverModel(config)
    model.load_state_dict(
        torch.load(ckpt_dir / "model.pt", map_location="cpu", weights_only=True),
    )
    model.to(device)
    model.eval()

    test_ds = BeatSaberDataset(Path(args.data), Path(args.audio_manifest), config, split="test")
    results = []

    for i in range(len(test_ds)):
        mel, tokens, mask = test_ds[i]
        sample = test_ds.samples[i]

        # Greedy decoding (temperature=0) degenerates into a repetition trap for
        # this model — it gets stuck predicting BAR/POS_EMPTY forever and never
        # places a note or reaches END. Sample instead (matching production
        # generate() defaults), with a per-sample fixed seed for reproducible
        # evaluation runs.
        gen_tokens = generate(model, mel, sample["difficulty"], config, temperature=1.0, seed=i)
        gen_notes = decode_tokens(gen_tokens, sample["bpm"], include_bombs=config.include_bombs)
        # Exclude decoded bomb entries (color=3) from both sides for an
        # apples-to-apples color-note comparison.
        gen_notes = [n for n in gen_notes if n.color in (0, 1)]

        # dataset.py frees the raw `notes` dicts after tokenization (memory
        # optimization for training) — reconstruct reference notes from the
        # already-kept token_ids instead of the (no longer present) raw notes.
        ref_notes = decode_tokens(sample["token_ids"], sample["bpm"], include_bombs=config.include_bombs)
        ref_notes = [n for n in ref_notes if n.color in (0, 1)]

        metrics = evaluate_map(gen_notes, ref_notes, sample["bpm"])
        metrics["song_hash"] = sample["song_hash"]
        metrics["difficulty"] = sample["difficulty"]
        results.append(metrics)

    output_path = Path(args.output) if args.output else Path("evaluation_results.json")
    output_path.write_text(_json.dumps(results, indent=2), encoding="utf-8")
    print(f"Evaluated {len(results)} maps. Results: {output_path}")


def cmd_analyze_obstacles(args: argparse.Namespace) -> None:
    from beat_weaver.model.obstacles import mine_obstacle_stats, save_obstacle_stats

    stats = mine_obstacle_stats(Path(args.data))
    save_obstacle_stats(stats, Path(args.output))

    print(f"{'Difficulty':<12} {'per_min':>8} {'dur_beats':>10} {'width':>6} {'crouch%':>8}")
    for difficulty, s in stats.items():
        print(
            f"{difficulty:<12} {s['per_minute']:>8.2f} {s['duration_beats']:>10.2f} "
            f"{s['width']:>6} {s['crouch_fraction'] * 100:>7.1f}%"
        )
    print(f"Saved to {args.output}")


def cmd_analyze_arcs(args: argparse.Namespace) -> None:
    from beat_weaver.model.arcs import mine_arc_stats, save_arc_stats

    stats = mine_arc_stats(Path(args.data))
    save_arc_stats(stats, Path(args.output))

    print(f"{'Difficulty':<12} {'per_min':>8} {'gap_min':>8} {'gap_max':>8} {'move_min':>9}")
    for difficulty, s in stats.items():
        print(
            f"{difficulty:<12} {s['per_minute']:>8.2f} {s['time_gap_min']:>8.2f} "
            f"{s['time_gap_max']:>8.2f} {s['movement_min']:>9.2f}"
        )
    print(f"Saved to {args.output}")


def cmd_run(args: argparse.Namespace) -> None:
    from beat_weaver.pipeline.batch import PipelineConfig, run_pipeline

    config = PipelineConfig(
        beat_saber_path=Path(args.beat_saber),
        raw_dir=Path(args.raw_dir),
        output_dir=Path(args.output),
        cache_dir=Path(args.cache_dir),
        include_local=not args.no_local,
        include_beatsaver=not args.no_beatsaver,
        include_official=not args.no_official,
        min_score=args.min_score,
        max_beatsaver_maps=args.max_maps,
    )
    result = run_pipeline(config)
    print(f"Done: {result.total_songs} songs, {result.total_beatmaps} beatmaps, {result.total_notes} notes")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="beat-weaver",
        description="Beat Weaver training data pipeline",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable debug logging"
    )
    sub = parser.add_subparsers(dest="command")

    # download
    dl = sub.add_parser("download", help="Download custom maps from BeatSaver")
    dl.add_argument("--min-score", type=float, default=0.75,
                     help="Minimum rating score 0.0-1.0 (default: 0.75)")
    dl.add_argument("--min-upvotes", type=int, default=5,
                     help="Minimum upvotes (default: 5)")
    dl.add_argument("--max-maps", type=int, default=0,
                     help="Max maps to download (default: 0 = unlimited)")
    dl.add_argument("--workers", type=int, default=8,
                     help="Parallel download threads (default: 8)")
    dl.add_argument("--difficulty", nargs="+", default=None,
                     choices=["Easy", "Normal", "Hard", "Expert", "ExpertPlus"],
                     help="Only download maps offering at least one of these "
                          "difficulties (default: no filter, any difficulty)")
    dl.add_argument("--output", default="data/raw/beatsaver")

    # extract-official
    ext = sub.add_parser("extract-official", help="Extract maps from Unity bundles")
    ext.add_argument(
        "--beat-saber",
        default=r"C:\Program Files (x86)\Steam\steamapps\common\Beat Saber",
    )
    ext.add_argument("--output", default="data/raw/official")

    # build-manifest
    bm = sub.add_parser("build-manifest", help="Build audio manifest from raw map folders")
    bm.add_argument("--input", nargs="+", default=["data/raw"],
                     help="Raw map directories to scan (default: data/raw)")
    bm.add_argument("--output", default="data/audio_manifest.json",
                     help="Output manifest JSON path")

    # process
    proc = sub.add_parser("process", help="Normalize raw maps into Parquet")
    proc.add_argument("--input", default="data/raw")
    proc.add_argument("--output", default="data/processed")
    proc.add_argument("--difficulty", nargs="+", default=None,
                     choices=["Easy", "Normal", "Hard", "Expert", "ExpertPlus"],
                     help="Only write these difficulties to Parquet, excluding all "
                          "others from the processed data entirely (default: no "
                          "filter, every difficulty in each map is included)")

    # run (full pipeline)
    run = sub.add_parser("run", help="Run full pipeline")
    run.add_argument(
        "--beat-saber",
        default=r"C:\Program Files (x86)\Steam\steamapps\common\Beat Saber",
    )
    run.add_argument("--raw-dir", default="data/raw")
    run.add_argument("--output", default="data/processed")
    run.add_argument("--cache-dir", default="data/cache")
    run.add_argument("--min-score", type=float, default=0.7)
    run.add_argument("--max-maps", type=int, default=100)
    run.add_argument("--no-local", action="store_true")
    run.add_argument("--no-beatsaver", action="store_true")
    run.add_argument("--no-official", action="store_true")

    # train
    tr = sub.add_parser("train", help="Train the ML model")
    tr.add_argument("--data", default="data/processed", help="Processed data directory")
    tr.add_argument("--audio-manifest", required=True, help="Audio manifest JSON path")
    tr.add_argument("--output", default="output/training", help="Output directory for checkpoints")
    tr.add_argument("--config", default=None, help="Optional JSON config override")
    tr.add_argument("--epochs", type=int, default=None, help="Max epochs")
    tr.add_argument("--batch-size", type=int, default=None, help="Batch size")
    tr.add_argument("--resume", default=None, help="Resume from checkpoint directory")

    # generate
    gen = sub.add_parser("generate", help="Generate a Beat Saber map from audio")
    gen.add_argument("--checkpoint", required=True, help="Model checkpoint directory")
    audio_source = gen.add_mutually_exclusive_group(required=True)
    audio_source.add_argument("--audio", default=None, help="Input audio file")
    audio_source.add_argument("--audio-dir", default=None,
                     help="Folder of audio files to generate one at a time (queue), "
                          "each into its own subfolder under --output")
    gen.add_argument("--difficulty", nargs="+", default=["Expert"],
                     choices=["Easy", "Normal", "Hard", "Expert", "ExpertPlus"],
                     help="One or more difficulties to generate into the same map folder")
    gen.add_argument("--output", default=None,
                     help="Output map folder (--audio), or base directory for per-song "
                          "subfolders (--audio-dir)")
    gen.add_argument("--bpm", type=float, default=None,
                     help="Song BPM (auto-detected from audio if not provided)")
    gen.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    gen.add_argument("--seed", type=int, default=None, help="Random seed")
    gen.add_argument("--no-obstacles", action="store_true",
                     help="Skip rule-based wall/obstacle placement")
    gen.add_argument("--obstacle-stats", default="data/processed/obstacle_stats.json",
                     help="Mined obstacle stats JSON from 'analyze-obstacles' "
                          "(falls back to built-in defaults if missing)")

    # analyze-obstacles
    ao = sub.add_parser("analyze-obstacles", help="Mine wall/obstacle placement stats from processed data")
    ao.add_argument("--data", default="data/processed", help="Processed data directory")
    ao.add_argument("--output", default="data/processed/obstacle_stats.json",
                    help="Output stats JSON path")

    # analyze-arcs
    aa = sub.add_parser("analyze-arcs", help="Mine arc/slider placement stats from processed data")
    aa.add_argument("--data", default="data/processed", help="Processed data directory")
    aa.add_argument("--output", default="data/processed/arc_stats.json",
                    help="Output stats JSON path")

    # evaluate
    ev = sub.add_parser("evaluate", help="Evaluate model on test data")
    ev.add_argument("--checkpoint", required=True, help="Model checkpoint directory")
    ev.add_argument("--data", default="data/processed", help="Test data directory")
    ev.add_argument("--audio-manifest", required=True, help="Audio manifest JSON path")
    ev.add_argument("--output", default=None, help="Output JSON path for results")

    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    commands = {
        "download": cmd_download,
        "extract-official": cmd_extract_official,
        "build-manifest": cmd_build_manifest,
        "process": cmd_process,
        "run": cmd_run,
        "train": cmd_train,
        "generate": cmd_generate,
        "analyze-obstacles": cmd_analyze_obstacles,
        "analyze-arcs": cmd_analyze_arcs,
        "evaluate": cmd_evaluate,
    }

    handler = commands.get(args.command)
    if handler:
        handler(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
