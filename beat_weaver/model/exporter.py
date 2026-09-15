"""Export token sequences to playable v2 Beat Saber map folders."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import soundfile as sf

from beat_weaver.model.tokenizer import decode_tokens
from beat_weaver.schemas.normalized import Note, Obstacle

# Obstacle height -> v2 `_type` (RESEARCH.md: 0=full-height wall, 1=crouch wall)
_FULL_HEIGHT_TYPE = 0
_CROUCH_HEIGHT_TYPE = 1
_CROUCH_HEIGHT_THRESHOLD = 4  # height <= this is a crouch wall

# Beat Saber only plays Ogg Vorbis audio for custom levels (.egg/.ogg). Any
# other source format (mp3, wav, flac, m4a, ...) must be transcoded, or the
# generated map silently fails to load in-game despite being otherwise valid.
_NATIVE_AUDIO_SUFFIXES = {".ogg", ".egg"}

# libsndfile's OGG/Vorbis encoder crashes the process outright (no Python
# exception — a native abort) when writing a very large buffer in one shot
# via sf.write(). Writing in modest-sized chunks through a SoundFile object
# avoids it entirely; verified against multi-minute real songs (11M+ frames).
_TRANSCODE_CHUNK_FRAMES = 200_000


def _write_song_audio(source_path: Path, output_dir: Path) -> str:
    """Copy or transcode the source audio into output_dir as Ogg Vorbis.

    Returns the resulting filename (relative to output_dir) for use as
    `_songFilename` in Info.dat.
    """
    source_path = Path(source_path)
    if source_path.suffix.lower() in _NATIVE_AUDIO_SUFFIXES:
        filename = f"song{source_path.suffix}"
        shutil.copy2(source_path, output_dir / filename)
        return filename

    filename = "song.ogg"
    data, samplerate = sf.read(str(source_path), dtype="float32", always_2d=False)
    channels = data.shape[1] if data.ndim == 2 else 1
    with sf.SoundFile(
        str(output_dir / filename), mode="w",
        samplerate=samplerate, channels=channels, format="OGG", subtype="VORBIS",
    ) as f:
        for i in range(0, len(data), _TRANSCODE_CHUNK_FRAMES):
            f.write(data[i:i + _TRANSCODE_CHUNK_FRAMES])
    return filename

# NJS lookup by difficulty
_NJS_TABLE = {
    "Easy": 10,
    "Normal": 10,
    "Hard": 12,
    "Expert": 16,
    "ExpertPlus": 18,
}

# Difficulty rank for Info.dat
_DIFFICULTY_RANK = {
    "Easy": 1,
    "Normal": 3,
    "Hard": 5,
    "Expert": 7,
    "ExpertPlus": 9,
}


def _build_info_dat(
    song_name: str,
    bpm: float,
    difficulty: str,
    audio_filename: str = "song.ogg",
) -> dict:
    """Build a v2 Info.dat structure."""
    njs = _NJS_TABLE.get(difficulty, 16)
    rank = _DIFFICULTY_RANK.get(difficulty, 7)

    return {
        "_version": "2.0.0",
        "_songName": song_name,
        "_songSubName": "",
        "_songAuthorName": "",
        "_levelAuthorName": "BeatWeaver AI",
        "_beatsPerMinute": bpm,
        "_songTimeOffset": 0,
        "_shuffle": 0,
        "_shufflePeriod": 0.5,
        "_previewStartTime": 12,
        "_previewDuration": 10,
        "_songFilename": audio_filename,
        "_coverImageFilename": "",
        "_environmentName": "DefaultEnvironment",
        "_difficultyBeatmapSets": [
            {
                "_beatmapCharacteristicName": "Standard",
                "_difficultyBeatmaps": [
                    {
                        "_difficulty": difficulty,
                        "_difficultyRank": rank,
                        "_beatmapFilename": f"{difficulty}.dat",
                        "_noteJumpMovementSpeed": njs,
                        "_noteJumpStartBeatOffset": 0,
                    }
                ],
            }
        ],
    }


def _build_difficulty_dat(
    notes: list, obstacles: list[Obstacle] | None = None, version: str = "2.0.0",
) -> dict:
    """Build a v2 difficulty .dat structure from decoded notes (+ optional obstacles).

    Bomb notes are just Note entries with color=3 (decode_tokens' convention),
    so they fall out of the same loop as color notes with no special-casing.
    """
    v2_notes = []
    for note in notes:
        v2_notes.append({
            "_time": note.beat,
            "_lineIndex": note.x,
            "_lineLayer": note.y,
            "_type": note.color,
            "_cutDirection": note.cut_direction,
        })

    v2_obstacles = []
    for obs in obstacles or []:
        obs_type = _CROUCH_HEIGHT_TYPE if obs.height <= _CROUCH_HEIGHT_THRESHOLD else _FULL_HEIGHT_TYPE
        v2_obstacles.append({
            "_time": obs.beat,
            "_lineIndex": obs.x,
            "_lineLayer": obs.y,
            "_type": obs_type,
            "_duration": obs.duration_beats,
            "_width": obs.width,
        })

    return {
        "_version": version,
        "_notes": sorted(v2_notes, key=lambda n: n["_time"]),
        "_obstacles": sorted(v2_obstacles, key=lambda o: o["_time"]),
        "_events": [],
    }


def export_map(
    token_ids: list[int],
    bpm: float,
    song_name: str,
    audio_path: Path,
    output_dir: Path,
    difficulty: str = "Expert",
    include_bombs: bool = False,
    obstacles: list[Obstacle] | None = None,
) -> Path:
    """Export a token sequence to a playable v2 Beat Saber map folder.

    Args:
        token_ids: Generated token sequence.
        bpm: Song BPM.
        song_name: Display name for the song.
        audio_path: Path to the audio file.
        output_dir: Where to create the map folder.
        difficulty: Difficulty name.
        include_bombs: Must match the model config used to generate token_ids.
        obstacles: Optional walls (e.g. from obstacles.generate_obstacles()) to
            include alongside the decoded notes/bombs.

    Returns:
        Path to the created map folder.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Decode tokens to notes
    notes = decode_tokens(token_ids, bpm, include_bombs=include_bombs)

    # Write audio as Ogg Vorbis (transcoding if the source isn't already .ogg/.egg —
    # Beat Saber won't load a custom level with any other audio format)
    audio_filename = _write_song_audio(audio_path, output_dir)

    # Write Info.dat
    info = _build_info_dat(song_name, bpm, difficulty, audio_filename)
    (output_dir / "Info.dat").write_text(json.dumps(info, indent=2), encoding="utf-8")

    # Write difficulty file
    diff_dat = _build_difficulty_dat(notes, obstacles)
    (output_dir / f"{difficulty}.dat").write_text(json.dumps(diff_dat, indent=2), encoding="utf-8")

    return output_dir


def export_notes(
    notes: list[Note],
    bpm: float,
    song_name: str,
    audio_path: Path,
    output_dir: Path,
    difficulty: str = "Expert",
    obstacles: list[Obstacle] | None = None,
) -> Path:
    """Export a note list to a playable v2 Beat Saber map folder.

    Unlike export_map() which takes token IDs, this takes pre-decoded notes
    (e.g. from windowed generation where notes have already been merged).

    Args:
        notes: List of Note objects (bomb entries are color=3, per decode_tokens).
        bpm: Song BPM.
        song_name: Display name for the song.
        audio_path: Path to the audio file.
        output_dir: Where to create the map folder.
        difficulty: Difficulty name.
        obstacles: Optional walls (e.g. from obstacles.generate_obstacles()).

    Returns:
        Path to the created map folder.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    audio_filename = _write_song_audio(audio_path, output_dir)

    info = _build_info_dat(song_name, bpm, difficulty, audio_filename)
    (output_dir / "Info.dat").write_text(json.dumps(info, indent=2), encoding="utf-8")

    diff_dat = _build_difficulty_dat(notes, obstacles)
    (output_dir / f"{difficulty}.dat").write_text(json.dumps(diff_dat, indent=2), encoding="utf-8")

    return output_dir
