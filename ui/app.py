"""Streamlit control panel for the beat-weaver CLI pipeline.

Each tab wraps one `beat-weaver` subcommand: it builds the same arguments the
CLI accepts, runs it as a background subprocess, and tails its log so
long-running stages (download, train) don't block the UI.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = PROJECT_ROOT / "output" / "ui_logs"
UPLOAD_DIR = PROJECT_ROOT / "output" / "ui_uploads"
LOG_DIR.mkdir(parents=True, exist_ok=True)

DIFFICULTIES = ["Easy", "Normal", "Hard", "Expert", "ExpertPlus"]
DEFAULT_BEAT_SABER = r"D:\Beat Saber"


def start_job(key: str, args: list[str]) -> None:
    log_path = LOG_DIR / f"{key}.log"
    log_file = open(log_path, "w", encoding="utf-8", errors="replace")
    proc = subprocess.Popen(
        [sys.executable, "-m", "beat_weaver.cli", *args],
        cwd=PROJECT_ROOT,
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )
    st.session_state[f"proc_{key}"] = proc
    st.session_state[f"logfile_{key}"] = log_file
    st.session_state[f"logpath_{key}"] = log_path


def stop_job(key: str) -> None:
    proc = st.session_state.get(f"proc_{key}")
    if proc is not None and proc.poll() is None:
        proc.terminate()


def job_running(key: str) -> bool:
    proc = st.session_state.get(f"proc_{key}")
    return proc is not None and proc.poll() is None


def job_status(key: str) -> str | None:
    proc = st.session_state.get(f"proc_{key}")
    if proc is None:
        return None
    code = proc.poll()
    if code is None:
        return "running"
    logfile = st.session_state.get(f"logfile_{key}")
    if logfile is not None and not logfile.closed:
        logfile.close()
    return "finished" if code == 0 else f"failed (exit code {code})"


def show_command_preview(args: list[str]) -> None:
    st.caption("Command: `" + " ".join(["beat-weaver", *args]) + "`")


@st.fragment(run_every=2)
def render_log_panel(key: str) -> None:
    status = job_status(key)
    if status is None:
        st.caption("Not started yet.")
        return
    col1, col2 = st.columns([1, 5])
    with col1:
        if st.button("Stop", key=f"stop_{key}", disabled=status != "running"):
            stop_job(key)
            st.rerun()
    with col2:
        st.caption(f"Status: {status}")
    log_path = st.session_state.get(f"logpath_{key}")
    if log_path and Path(log_path).exists():
        text = Path(log_path).read_text(encoding="utf-8", errors="replace")
        st.code(text[-6000:] or "(no output yet)", language="text")


def zip_directory(directory: Path) -> Path:
    archive_base = LOG_DIR / directory.name
    archive_path = shutil.make_archive(str(archive_base), "zip", root_dir=directory)
    return Path(archive_path)


def tab_generate() -> None:
    st.subheader("Generate a map from audio")
    checkpoint = st.text_input("Checkpoint directory", "output/training/checkpoints/best", key="gen_ckpt")

    source_mode = st.radio(
        "Audio source", ["Single file", "Folder (queue)"], horizontal=True, key="gen_source_mode",
    )
    audio_file = None
    audio_dir = None
    if source_mode == "Single file":
        audio_file = st.file_uploader(
            "Audio file", type=["wav", "ogg", "mp3", "flac", "m4a"], key="gen_audio"
        )
    else:
        audio_dir = st.text_input(
            "Folder of audio files (on disk — processed one at a time)",
            "", key="gen_audio_dir",
        )
        if audio_dir and not Path(audio_dir).is_dir():
            st.caption(":red[Folder not found]")

    st.caption("Difficulties (check one or more — all share one map folder)")
    diff_cols = st.columns(len(DIFFICULTIES))
    selected_difficulties = []
    for col, diff_name in zip(diff_cols, DIFFICULTIES):
        with col:
            if st.checkbox(diff_name, value=(diff_name == "Expert"), key=f"gen_diff_{diff_name}"):
                selected_difficulties.append(diff_name)
    auto_bpm = st.checkbox("Auto-detect BPM", value=True, key="gen_auto_bpm")
    bpm = None
    if not auto_bpm:
        bpm = st.number_input("BPM", min_value=1.0, value=120.0, key="gen_bpm")
    temperature = st.slider("Temperature", 0.1, 2.0, 1.0, key="gen_temp")
    use_seed = st.checkbox("Fix random seed", value=False, key="gen_use_seed")
    seed = None
    if use_seed:
        seed = st.number_input("Seed", min_value=0, value=0, step=1, key="gen_seed")
    output_label = (
        "Output base folder — one subfolder per song (blank = auto)"
        if source_mode == "Folder (queue)"
        else "Output folder (blank = auto)"
    )
    output_dir = st.text_input(output_label, "", key="gen_output")
    add_obstacles = st.checkbox("Add walls/obstacles (rule-based)", value=True, key="gen_add_obstacles")
    obstacle_stats = st.text_input(
        "Obstacle stats JSON (blank = built-in defaults)",
        "data/processed/obstacle_stats.json", key="gen_obstacle_stats",
    )

    args = ["generate", "--checkpoint", checkpoint, "--difficulty", *selected_difficulties,
            "--temperature", str(temperature)]
    if bpm is not None:
        args += ["--bpm", str(bpm)]
    if seed is not None:
        args += ["--seed", str(int(seed))]
    if output_dir:
        args += ["--output", output_dir]
    if not add_obstacles:
        args.append("--no-obstacles")
    elif obstacle_stats:
        args += ["--obstacle-stats", obstacle_stats]
    if source_mode == "Folder (queue)":
        show_command_preview([*args, "--audio-dir", audio_dir or "<folder>"])
    else:
        show_command_preview([*args, "--audio", audio_file.name if audio_file else "<audio file>"])
    if not selected_difficulties:
        st.caption(":red[Check at least one difficulty]")

    if source_mode == "Folder (queue)":
        can_run = bool(audio_dir) and Path(audio_dir).is_dir() and bool(selected_difficulties) and not job_running("generate")
    else:
        can_run = audio_file is not None and bool(selected_difficulties) and not job_running("generate")

    if st.button("Generate", key="gen_start", disabled=not can_run):
        if source_mode == "Folder (queue)":
            run_args = [*args, "--audio-dir", audio_dir]
            st.session_state["gen_last_output"] = output_dir or "output"
            st.session_state["gen_last_is_queue"] = True
        else:
            UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
            audio_path = UPLOAD_DIR / audio_file.name
            audio_path.write_bytes(audio_file.getvalue())
            run_args = [*args, "--audio", str(audio_path)]
            st.session_state["gen_last_output"] = output_dir or f"output/{audio_path.stem}"
            st.session_state["gen_last_is_queue"] = False
        start_job("generate", run_args)
        st.rerun()

    render_log_panel("generate")

    status = job_status("generate")
    if status == "finished":
        out_dir = PROJECT_ROOT / st.session_state.get("gen_last_output", "")
        if st.session_state.get("gen_last_is_queue"):
            st.caption(f"Generated maps are in `{out_dir}` — one subfolder per song.")
        elif out_dir.exists():
            zip_path = zip_directory(out_dir)
            st.download_button(
                "Download generated map (.zip)",
                zip_path.read_bytes(),
                file_name=f"{out_dir.name}.zip",
                key="gen_dl",
            )


def tab_train() -> None:
    st.subheader("Train the ML model")
    config_files = ["(default)"] + sorted(p.name for p in (PROJECT_ROOT / "configs").glob("*.json"))
    config_choice = st.selectbox("Config", config_files, key="tr_config")
    data_dir = st.text_input("Processed data directory", "data/processed", key="tr_data")
    manifest = st.text_input("Audio manifest path", "data/audio_manifest.json", key="tr_manifest")
    output_dir = st.text_input("Checkpoint output directory", "output/training", key="tr_output")
    epochs = st.number_input("Max epochs (0 = use config default)", 0, value=0, key="tr_epochs")
    batch_size = st.number_input("Batch size (0 = use config default)", 0, value=0, key="tr_batch")
    resume = st.text_input("Resume from checkpoint dir (optional)", "", key="tr_resume")

    args = ["train", "--data", data_dir, "--audio-manifest", manifest, "--output", output_dir]
    if config_choice != "(default)":
        args += ["--config", f"configs/{config_choice}"]
    if epochs:
        args += ["--epochs", str(int(epochs))]
    if batch_size:
        args += ["--batch-size", str(int(batch_size))]
    if resume:
        args += ["--resume", resume]
    show_command_preview(args)

    if st.button("Start training", key="tr_start", disabled=job_running("train")):
        start_job("train", args)
        st.rerun()

    render_log_panel("train")


def tab_evaluate() -> None:
    st.subheader("Evaluate a model checkpoint")
    checkpoint = st.text_input("Checkpoint directory", "output/training/checkpoints/best", key="ev_ckpt")
    data_dir = st.text_input("Processed data directory", "data/processed", key="ev_data")
    manifest = st.text_input("Audio manifest path", "data/audio_manifest.json", key="ev_manifest")
    output = st.text_input("Output JSON path (optional)", "", key="ev_output")

    args = ["evaluate", "--checkpoint", checkpoint, "--data", data_dir, "--audio-manifest", manifest]
    if output:
        args += ["--output", output]
    show_command_preview(args)

    if st.button("Run evaluation", key="ev_start", disabled=job_running("evaluate")):
        start_job("evaluate", args)
        st.rerun()

    render_log_panel("evaluate")


def tab_analyze_obstacles() -> None:
    st.subheader("Mine wall/obstacle placement stats from processed data")
    st.caption("Run after Process — grounds Generate's rule-based wall placement in real map data instead of built-in defaults.")
    data_dir = st.text_input("Processed data directory", "data/processed", key="ao_data")
    output = st.text_input("Output stats JSON path", "data/processed/obstacle_stats.json", key="ao_output")

    args = ["analyze-obstacles", "--data", data_dir, "--output", output]
    show_command_preview(args)

    if st.button("Analyze obstacles", key="ao_start", disabled=job_running("analyze-obstacles")):
        start_job("analyze-obstacles", args)
        st.rerun()

    render_log_panel("analyze-obstacles")


def tab_download() -> None:
    st.subheader("Download custom maps from BeatSaver")
    min_score = st.slider("Min score", 0.0, 1.0, 0.75, key="dl_min_score")
    min_upvotes = st.number_input("Min upvotes", 0, value=5, key="dl_min_upvotes")
    max_maps = st.number_input("Max maps (0 = unlimited)", 0, value=0, key="dl_max_maps")
    workers = st.number_input("Parallel workers", 1, value=8, key="dl_workers")
    output = st.text_input("Output directory", "data/raw/beatsaver", key="dl_output")

    st.caption("Difficulties (check one or more — blank = no filter, any difficulty)")
    diff_cols = st.columns(len(DIFFICULTIES))
    selected_difficulties = []
    for col, diff_name in zip(diff_cols, DIFFICULTIES):
        with col:
            if st.checkbox(diff_name, value=False, key=f"dl_diff_{diff_name}"):
                selected_difficulties.append(diff_name)

    args = ["download", "--min-score", str(min_score), "--min-upvotes", str(int(min_upvotes)),
            "--max-maps", str(int(max_maps)), "--workers", str(int(workers)), "--output", output]
    if selected_difficulties:
        args += ["--difficulty", *selected_difficulties]
    show_command_preview(args)

    if st.button("Start download", key="dl_start", disabled=job_running("download")):
        start_job("download", args)
        st.rerun()

    render_log_panel("download")


def tab_extract_official() -> None:
    st.subheader("Extract official maps from Beat Saber's Unity bundles")
    beat_saber = st.text_input("Beat Saber install path", DEFAULT_BEAT_SABER, key="eo_path")
    output = st.text_input("Output directory", "data/raw/official", key="eo_output")

    args = ["extract-official", "--beat-saber", beat_saber, "--output", output]
    show_command_preview(args)

    if st.button("Start extraction", key="eo_start", disabled=job_running("extract-official")):
        start_job("extract-official", args)
        st.rerun()

    render_log_panel("extract-official")


def tab_build_manifest() -> None:
    st.subheader("Build an audio manifest from raw map folders")
    inputs = st.text_input("Raw map directories (comma-separated)", "data/raw", key="bm_input")
    output = st.text_input("Output manifest path", "data/audio_manifest.json", key="bm_output")

    input_dirs = [s.strip() for s in inputs.split(",") if s.strip()]
    args = ["build-manifest", "--input", *input_dirs, "--output", output]
    show_command_preview(args)

    if st.button("Build manifest", key="bm_start", disabled=job_running("build-manifest")):
        start_job("build-manifest", args)
        st.rerun()

    render_log_panel("build-manifest")


def tab_process() -> None:
    st.subheader("Normalize raw maps into Parquet")
    input_dir = st.text_input("Raw maps directory", "data/raw", key="pr_input")
    output_dir = st.text_input("Output directory", "data/processed", key="pr_output")

    st.caption("Difficulties to keep (check one or more — blank = no filter, every difficulty is written to Parquet)")
    diff_cols = st.columns(len(DIFFICULTIES))
    selected_difficulties = []
    for col, diff_name in zip(diff_cols, DIFFICULTIES):
        with col:
            if st.checkbox(diff_name, value=False, key=f"pr_diff_{diff_name}"):
                selected_difficulties.append(diff_name)

    args = ["process", "--input", input_dir, "--output", output_dir]
    if selected_difficulties:
        args += ["--difficulty", *selected_difficulties]
    show_command_preview(args)

    if st.button("Start processing", key="pr_start", disabled=job_running("process")):
        start_job("process", args)
        st.rerun()

    render_log_panel("process")


def tab_run_pipeline() -> None:
    st.subheader("Run the full pipeline (download + extract + process)")
    beat_saber = st.text_input("Beat Saber install path", DEFAULT_BEAT_SABER, key="rn_bs")
    raw_dir = st.text_input("Raw directory", "data/raw", key="rn_raw")
    output_dir = st.text_input("Output directory", "data/processed", key="rn_output")
    cache_dir = st.text_input("Cache directory", "data/cache", key="rn_cache")
    min_score = st.slider("Min score", 0.0, 1.0, 0.7, key="rn_min_score")
    max_maps = st.number_input("Max maps", 0, value=100, key="rn_max_maps")
    no_local = st.checkbox("Skip local custom maps", key="rn_no_local")
    no_beatsaver = st.checkbox("Skip BeatSaver download", key="rn_no_bs")
    no_official = st.checkbox("Skip official maps", key="rn_no_official")

    args = ["run", "--beat-saber", beat_saber, "--raw-dir", raw_dir, "--output", output_dir,
            "--cache-dir", cache_dir, "--min-score", str(min_score), "--max-maps", str(int(max_maps))]
    if no_local:
        args.append("--no-local")
    if no_beatsaver:
        args.append("--no-beatsaver")
    if no_official:
        args.append("--no-official")
    show_command_preview(args)

    if st.button("Run full pipeline", key="rn_start", disabled=job_running("run")):
        start_job("run", args)
        st.rerun()

    render_log_panel("run")


def main() -> None:
    st.set_page_config(page_title="Beat Weaver", layout="wide")
    st.title("Beat Weaver")
    st.caption("Local control panel for the beat-weaver CLI pipeline")

    tabs = st.tabs([
        "Generate", "Train", "Evaluate", "Analyze Obstacles", "Download",
        "Extract Official", "Build Manifest", "Process", "Full Pipeline",
    ])
    with tabs[0]:
        tab_generate()
    with tabs[1]:
        tab_train()
    with tabs[2]:
        tab_evaluate()
    with tabs[3]:
        tab_analyze_obstacles()
    with tabs[4]:
        tab_download()
    with tabs[5]:
        tab_extract_official()
    with tabs[6]:
        tab_build_manifest()
    with tabs[7]:
        tab_process()
    with tabs[8]:
        tab_run_pipeline()


if __name__ == "__main__":
    main()
