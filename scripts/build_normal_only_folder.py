"""One-off data-prep script for the Normal-only training experiment.

Non-destructively builds a trimmed copy of each source map folder
containing only: the song audio, Info.dat (rewritten to list only the
Standard-characteristic Normal difficulty), and that Normal .dat file
itself. Originals in the source directories are never modified or deleted.

Usage:
    python scripts/build_normal_only_folder.py <dest_dir> <src_dir> [<src_dir> ...]
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path


def _is_v2_info(info: dict) -> bool:
    """Same detection rule as beat_weaver.parsers.info_parser.parse_info()."""
    return "_version" in info or "_songName" in info


def _find_standard_normal_v2(info: dict) -> tuple[dict | None, str]:
    """Return (beatmap entry, audio filename) for v2-style Info.dat."""
    for bset in info.get("_difficultyBeatmapSets", []):
        if bset.get("_beatmapCharacteristicName") != "Standard":
            continue
        for bmap in bset.get("_difficultyBeatmaps", []):
            if bmap.get("_difficulty") == "Normal":
                return bmap, info.get("_songFilename", "")
    return None, ""


def _find_standard_normal_v4(info: dict) -> tuple[dict | None, str]:
    """Return (beatmap entry, audio filename) for raw v4-style Info.dat."""
    for bmap in info.get("difficultyBeatmaps", []):
        if bmap.get("characteristic") == "Standard" and bmap.get("difficulty") == "Normal":
            return bmap, info.get("audio", {}).get("songFilename", "")
    return None, ""


def _trim_info_v2(info: dict, normal_entry: dict) -> dict:
    trimmed = dict(info)
    trimmed["_difficultyBeatmapSets"] = [
        {"_beatmapCharacteristicName": "Standard", "_difficultyBeatmaps": [normal_entry]}
    ]
    return trimmed


def _trim_info_v4(info: dict, normal_entry: dict) -> dict:
    trimmed = dict(info)
    trimmed["difficultyBeatmaps"] = [normal_entry]
    return trimmed


def build_normal_only_folder(dest_dir: Path, src_dirs: list[Path]) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    kept = 0
    skipped_no_normal = 0
    skipped_missing_files = 0

    for src_dir in src_dirs:
        for info_path in sorted(src_dir.rglob("Info.dat")):
            map_folder = info_path.parent
            info = json.loads(info_path.read_text(encoding="utf-8"))

            is_v2 = _is_v2_info(info)
            if is_v2:
                normal_entry, audio_filename = _find_standard_normal_v2(info)
                beatmap_filename = normal_entry.get("_beatmapFilename", "") if normal_entry else ""
            else:
                normal_entry, audio_filename = _find_standard_normal_v4(info)
                beatmap_filename = normal_entry.get("beatmapDataFilename", "") if normal_entry else ""

            if normal_entry is None:
                skipped_no_normal += 1
                continue

            beatmap_src = map_folder / beatmap_filename
            audio_src = map_folder / audio_filename

            if not beatmap_src.exists() or not audio_src.exists():
                skipped_missing_files += 1
                continue

            out_folder = dest_dir / map_folder.name
            out_folder.mkdir(parents=True, exist_ok=True)

            shutil.copy2(audio_src, out_folder / audio_filename)
            shutil.copy2(beatmap_src, out_folder / beatmap_filename)

            trimmed_info = _trim_info_v2(info, normal_entry) if is_v2 else _trim_info_v4(info, normal_entry)
            (out_folder / "Info.dat").write_text(
                json.dumps(trimmed_info, indent=2), encoding="utf-8"
            )

            kept += 1

    print(f"Built {kept} Normal-only map folders in {dest_dir}")
    print(f"  Skipped (no Standard/Normal difficulty): {skipped_no_normal}")
    print(f"  Skipped (referenced file missing): {skipped_missing_files}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    dest = Path(sys.argv[1])
    sources = [Path(p) for p in sys.argv[2:]]
    build_normal_only_folder(dest, sources)
