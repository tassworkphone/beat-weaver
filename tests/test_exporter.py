"""Tests for the v2 Beat Saber map exporter."""

import json

import pytest

np = pytest.importorskip("numpy")
sf = pytest.importorskip("soundfile")

from beat_weaver.model.exporter import export_map, export_multi_difficulty, export_notes
from beat_weaver.model.tokenizer import (
    BAR,
    DIFF_EXPERT,
    END,
    LEFT_EMPTY,
    POS_BASE,
    RIGHT_EMPTY,
    START,
    _encode_bomb_token,
    _encode_note_token,
    LEFT_BASE,
    RIGHT_BASE,
)
from beat_weaver.schemas.normalized import Note, Obstacle


@pytest.fixture
def audio_file(tmp_path):
    """Create a dummy audio file."""
    import soundfile as sf

    sr = 22050
    audio = np.zeros(sr * 2, dtype=np.float32)
    path = tmp_path / "song.ogg"
    sf.write(str(path), audio, sr)
    return path


@pytest.fixture
def wav_audio_file(tmp_path):
    """A non-Ogg source file — exercises the transcode path."""
    sr = 22050
    audio = np.zeros(sr * 2, dtype=np.float32)
    path = tmp_path / "song.wav"
    sf.write(str(path), audio, sr)
    return path


class TestExportMap:
    def test_creates_folder_structure(self, tmp_path, audio_file):
        # Simple token sequence: one left note at beat 0
        tokens = [
            START, DIFF_EXPERT, BAR,
            POS_BASE + 0,
            _encode_note_token(LEFT_BASE, 1, 0, 1),
            RIGHT_EMPTY,
            END,
        ]
        output = tmp_path / "output_map"
        result = export_map(tokens, bpm=120.0, song_name="Test Song",
                           audio_path=audio_file, output_dir=output)

        assert (result / "Info.dat").exists()
        assert (result / "Expert.dat").exists()
        assert (result / "song.ogg").exists()

    def test_info_dat_structure(self, tmp_path, audio_file):
        tokens = [START, DIFF_EXPERT, BAR, POS_BASE, LEFT_EMPTY, RIGHT_EMPTY, END]
        output = tmp_path / "output_map"
        export_map(tokens, bpm=128.0, song_name="My Song",
                  audio_path=audio_file, output_dir=output)

        info = json.loads((output / "Info.dat").read_text())
        assert info["_version"] == "2.0.0"
        assert info["_songName"] == "My Song"
        assert info["_beatsPerMinute"] == 128.0
        assert info["_levelAuthorName"] == "BeatWeaver AI"

        sets = info["_difficultyBeatmapSets"]
        assert len(sets) == 1
        assert sets[0]["_beatmapCharacteristicName"] == "Standard"
        bm = sets[0]["_difficultyBeatmaps"][0]
        assert bm["_difficulty"] == "Expert"
        assert bm["_noteJumpMovementSpeed"] == 16

    def test_difficulty_dat_notes(self, tmp_path, audio_file):
        tokens = [
            START, DIFF_EXPERT, BAR,
            POS_BASE + 0,
            _encode_note_token(LEFT_BASE, 2, 1, 3),
            _encode_note_token(RIGHT_BASE, 1, 0, 1),
            END,
        ]
        output = tmp_path / "output_map"
        export_map(tokens, bpm=120.0, song_name="Test",
                  audio_path=audio_file, output_dir=output)

        dat = json.loads((output / "Expert.dat").read_text())
        assert dat["_version"] == "2.0.0"
        notes = dat["_notes"]
        assert len(notes) == 2

        # Left note (color=0)
        left = [n for n in notes if n["_type"] == 0][0]
        assert left["_lineIndex"] == 2
        assert left["_lineLayer"] == 1
        assert left["_cutDirection"] == 3

        # Right note (color=1)
        right = [n for n in notes if n["_type"] == 1][0]
        assert right["_lineIndex"] == 1
        assert right["_lineLayer"] == 0
        assert right["_cutDirection"] == 1

    def test_empty_map(self, tmp_path, audio_file):
        tokens = [START, DIFF_EXPERT, END]
        output = tmp_path / "output_map"
        export_map(tokens, bpm=120.0, song_name="Empty",
                  audio_path=audio_file, output_dir=output)

        dat = json.loads((output / "Expert.dat").read_text())
        assert dat["_notes"] == []


class TestExportNotes:
    def test_export_notes_creates_valid_map(self, tmp_path, audio_file):
        """export_notes produces valid v2 map files from a note list."""
        notes = [
            Note(beat=0.0, time_seconds=0.0, x=1, y=0, color=0, cut_direction=1),
            Note(beat=0.5, time_seconds=0.25, x=2, y=1, color=1, cut_direction=0),
            Note(beat=4.0, time_seconds=2.0, x=3, y=2, color=0, cut_direction=3),
        ]
        output = tmp_path / "notes_map"
        result = export_notes(notes, bpm=120.0, song_name="Test Notes",
                              audio_path=audio_file, output_dir=output)

        assert (result / "Info.dat").exists()
        assert (result / "Expert.dat").exists()
        assert (result / "song.ogg").exists()

        info = json.loads((result / "Info.dat").read_text())
        assert info["_songName"] == "Test Notes"
        assert info["_beatsPerMinute"] == 120.0

        dat = json.loads((result / "Expert.dat").read_text())
        assert len(dat["_notes"]) == 3
        # Notes should be sorted by time
        times = [n["_time"] for n in dat["_notes"]]
        assert times == sorted(times)
        # Check first note
        assert dat["_notes"][0]["_lineIndex"] == 1
        assert dat["_notes"][0]["_type"] == 0
        assert dat["_notes"][0]["_cutDirection"] == 1

    def test_export_notes_empty(self, tmp_path, audio_file):
        """export_notes handles an empty note list."""
        output = tmp_path / "empty_notes_map"
        result = export_notes([], bpm=120.0, song_name="Empty",
                              audio_path=audio_file, output_dir=output)

        dat = json.loads((result / "Expert.dat").read_text())
        assert dat["_notes"] == []


class TestAudioTranscoding:
    def test_ogg_source_copied_as_is(self, tmp_path, audio_file):
        """An already-Ogg source is copied verbatim (fast path, no re-encode)."""
        output = tmp_path / "ogg_map"
        result = export_notes([], bpm=120.0, song_name="Ogg Test",
                              audio_path=audio_file, output_dir=output)
        assert (result / "song.ogg").exists()
        info = json.loads((result / "Info.dat").read_text())
        assert info["_songFilename"] == "song.ogg"

    def test_non_ogg_source_transcoded_to_ogg(self, tmp_path, wav_audio_file):
        """A .wav source must be transcoded — Beat Saber can't play song.wav."""
        output = tmp_path / "wav_map"
        result = export_notes([], bpm=120.0, song_name="Wav Test",
                              audio_path=wav_audio_file, output_dir=output)

        assert (result / "song.ogg").exists()
        assert not (result / "song.wav").exists()

        info = json.loads((result / "Info.dat").read_text())
        assert info["_songFilename"] == "song.ogg"

        # The written file must actually be valid, readable Ogg Vorbis.
        data, sr = sf.read(str(result / "song.ogg"))
        assert sr == 22050
        assert len(data) > 0

    def test_egg_source_copied_as_is(self, tmp_path):
        """.egg (Beat Saber's own extension for Ogg Vorbis) also hits the fast path."""
        sr = 22050
        audio = np.zeros(sr, dtype=np.float32)
        egg_path = tmp_path / "song.egg"
        sf.write(str(egg_path), audio, sr, format="OGG", subtype="VORBIS")

        output = tmp_path / "egg_map"
        result = export_notes([], bpm=120.0, song_name="Egg Test",
                              audio_path=egg_path, output_dir=output)
        assert (result / "song.egg").exists()

    def test_large_source_does_not_crash_encoder(self, tmp_path):
        """Regression: libsndfile's OGG/Vorbis encoder crashes the whole process
        (no Python exception) when writing a large buffer in a single sf.write()
        call — reproduced with a real ~4min song, min repro is a few million
        stereo frames. _write_song_audio must write in chunks to avoid this."""
        sr = 44100
        n_frames = 2_000_000  # well past the size that crashes single-shot write
        audio = np.zeros((n_frames, 2), dtype=np.float32)
        wav_path = tmp_path / "big.wav"
        sf.write(str(wav_path), audio, sr)

        output = tmp_path / "big_map"
        result = export_notes([], bpm=120.0, song_name="Big Test",
                              audio_path=wav_path, output_dir=output)

        ogg_path = result / "song.ogg"
        assert ogg_path.exists()
        info = sf.info(str(ogg_path))
        assert info.samplerate == sr
        assert info.channels == 2
        assert info.frames == n_frames


class TestExportBombs:
    def test_export_map_bomb_token_becomes_type_3_note(self, tmp_path, audio_file):
        """A generated BOMB token decodes to a _notes entry with _type: 3."""
        tokens = [
            START, DIFF_EXPERT, BAR,
            POS_BASE + 0,
            LEFT_EMPTY, RIGHT_EMPTY,
            _encode_bomb_token(2, 1),
            END,
        ]
        output = tmp_path / "bomb_map"
        export_map(tokens, bpm=120.0, song_name="Bomb Test",
                  audio_path=audio_file, output_dir=output, include_bombs=True)

        dat = json.loads((output / "Expert.dat").read_text())
        bombs = [n for n in dat["_notes"] if n["_type"] == 3]
        assert len(bombs) == 1
        assert bombs[0]["_lineIndex"] == 2
        assert bombs[0]["_lineLayer"] == 1

    def test_export_notes_bomb_note_object(self, tmp_path, audio_file):
        """export_notes writes a color=3 Note as a _type: 3 bomb entry."""
        notes = [
            Note(beat=0.0, time_seconds=0.0, x=1, y=0, color=0, cut_direction=1),
            Note(beat=0.0, time_seconds=0.0, x=3, y=2, color=3, cut_direction=0),
        ]
        output = tmp_path / "bomb_notes_map"
        result = export_notes(notes, bpm=120.0, song_name="Bomb Notes",
                              audio_path=audio_file, output_dir=output)

        dat = json.loads((result / "Expert.dat").read_text())
        assert len(dat["_notes"]) == 2
        bomb = next(n for n in dat["_notes"] if n["_type"] == 3)
        assert bomb["_lineIndex"] == 3
        assert bomb["_lineLayer"] == 2


class TestExportObstacles:
    def test_export_notes_writes_obstacles(self, tmp_path, audio_file):
        notes = [Note(beat=0.0, time_seconds=0.0, x=1, y=0, color=0, cut_direction=1)]
        obstacles = [
            Obstacle(beat=2.0, time_seconds=1.0, duration_beats=2.0, x=0, y=0, width=1, height=5),
            Obstacle(beat=6.0, time_seconds=3.0, duration_beats=1.0, x=2, y=0, width=1, height=3),
        ]
        output = tmp_path / "obstacle_map"
        result = export_notes(notes, bpm=120.0, song_name="Wall Test",
                              audio_path=audio_file, output_dir=output, obstacles=obstacles)

        dat = json.loads((result / "Expert.dat").read_text())
        assert len(dat["_obstacles"]) == 2

        full_height = next(o for o in dat["_obstacles"] if o["_lineIndex"] == 0)
        assert full_height["_type"] == 0  # full-height wall
        assert full_height["_duration"] == 2.0
        assert full_height["_width"] == 1

        crouch = next(o for o in dat["_obstacles"] if o["_lineIndex"] == 2)
        assert crouch["_type"] == 1  # crouch wall

    def test_export_notes_no_obstacles_defaults_empty(self, tmp_path, audio_file):
        notes = [Note(beat=0.0, time_seconds=0.0, x=1, y=0, color=0, cut_direction=1)]
        output = tmp_path / "no_obstacle_map"
        result = export_notes(notes, bpm=120.0, song_name="No Walls",
                              audio_path=audio_file, output_dir=output)
        dat = json.loads((result / "Expert.dat").read_text())
        assert dat["_obstacles"] == []


class TestExportMultiDifficulty:
    def test_writes_one_dat_per_difficulty(self, tmp_path, audio_file):
        maps = {
            "Normal": ([Note(beat=0.0, time_seconds=0.0, x=0, y=0, color=0, cut_direction=1)], []),
            "Expert": ([Note(beat=0.0, time_seconds=0.0, x=1, y=0, color=1, cut_direction=0)], []),
            "ExpertPlus": ([Note(beat=0.0, time_seconds=0.0, x=2, y=0, color=0, cut_direction=0)], []),
        }
        output = tmp_path / "multi_map"
        result = export_multi_difficulty(maps, bpm=120.0, song_name="Multi Test",
                                         audio_path=audio_file, output_dir=output)

        assert (result / "song.ogg").exists()
        for difficulty in maps:
            assert (result / f"{difficulty}.dat").exists()
        # Only one Info.dat / one audio file shared across all difficulties
        assert len(list(result.glob("*.ogg"))) == 1
        assert len(list(result.glob("Info.dat"))) == 1

    def test_info_dat_lists_all_difficulties_in_standard_order(self, tmp_path, audio_file):
        """Difficulties should be listed in standard order regardless of dict insertion order."""
        maps = {
            "ExpertPlus": ([], []),
            "Easy": ([], []),
            "Hard": ([], []),
        }
        output = tmp_path / "multi_order_map"
        result = export_multi_difficulty(maps, bpm=128.0, song_name="Order Test",
                                         audio_path=audio_file, output_dir=output)

        info = json.loads((result / "Info.dat").read_text())
        beatmaps = info["_difficultyBeatmapSets"][0]["_difficultyBeatmaps"]
        names = [b["_difficulty"] for b in beatmaps]
        assert names == ["Easy", "Hard", "ExpertPlus"]
        assert {b["_beatmapFilename"] for b in beatmaps} == {"Easy.dat", "Hard.dat", "ExpertPlus.dat"}

    def test_each_difficulty_dat_has_correct_notes(self, tmp_path, audio_file):
        maps = {
            "Easy": ([Note(beat=1.0, time_seconds=0.5, x=0, y=0, color=0, cut_direction=1)], []),
            "Expert": (
                [
                    Note(beat=0.0, time_seconds=0.0, x=1, y=0, color=0, cut_direction=1),
                    Note(beat=0.5, time_seconds=0.25, x=2, y=0, color=1, cut_direction=0),
                ],
                [],
            ),
        }
        output = tmp_path / "multi_notes_map"
        result = export_multi_difficulty(maps, bpm=120.0, song_name="Notes Test",
                                         audio_path=audio_file, output_dir=output)

        easy_dat = json.loads((result / "Easy.dat").read_text())
        expert_dat = json.loads((result / "Expert.dat").read_text())
        assert len(easy_dat["_notes"]) == 1
        assert len(expert_dat["_notes"]) == 2

    def test_requires_at_least_one_difficulty(self, tmp_path, audio_file):
        with pytest.raises(ValueError):
            export_multi_difficulty({}, bpm=120.0, song_name="Empty",
                                    audio_path=audio_file, output_dir=tmp_path / "empty_map")

    def test_njs_matches_each_difficulty(self, tmp_path, audio_file):
        maps = {"Easy": ([], []), "ExpertPlus": ([], [])}
        output = tmp_path / "njs_map"
        result = export_multi_difficulty(maps, bpm=120.0, song_name="NJS Test",
                                         audio_path=audio_file, output_dir=output)

        info = json.loads((result / "Info.dat").read_text())
        beatmaps = {b["_difficulty"]: b for b in info["_difficultyBeatmapSets"][0]["_difficultyBeatmaps"]}
        assert beatmaps["Easy"]["_noteJumpMovementSpeed"] == 10
        assert beatmaps["ExpertPlus"]["_noteJumpMovementSpeed"] == 18
