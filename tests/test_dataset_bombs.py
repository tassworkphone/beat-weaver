"""Tests for BeatSaberDataset's opt-in bomb loading (config.include_bombs)."""

import json

import pytest

np = pytest.importorskip("numpy")
pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

from beat_weaver.model.config import ModelConfig
from beat_weaver.model.dataset import BeatSaberDataset
from beat_weaver.model.tokenizer import BOMB_BASE, BOMB_COUNT, BOMB_EMPTY


def _write_test_song(tmp_path, song_hash="hash_0001", bpm=120.0, with_bombs=True):
    from beat_weaver.storage.writer import BOMBS_SCHEMA, NOTES_SCHEMA

    processed = tmp_path / "processed"
    processed.mkdir()

    notes_data = {
        "song_hash": [song_hash] * 2,
        "source": ["beatsaver"] * 2,
        "difficulty": ["Expert"] * 2,
        "characteristic": ["Standard"] * 2,
        "bpm": [bpm] * 2,
        "beat": [0.0, 1.0],
        "time_seconds": [0.0, 0.5],
        "x": [1, 2],
        "y": [0, 0],
        "color": [0, 1],
        "cut_direction": [1, 1],
        "angle_offset": [0, 0],
    }
    pq.write_table(pa.table(notes_data, schema=NOTES_SCHEMA), processed / "notes_0000.parquet")

    if with_bombs:
        bombs_data = {
            "song_hash": [song_hash],
            "source": ["beatsaver"],
            "difficulty": ["Expert"],
            "characteristic": ["Standard"],
            "bpm": [bpm],
            "beat": [2.0],  # a beat with no note — bomb-only position
            "time_seconds": [1.0],
            "x": [3],
            "y": [2],
        }
        pq.write_table(pa.table(bombs_data, schema=BOMBS_SCHEMA), processed / "bombs_0000.parquet")

    metadata = [{"hash": song_hash, "source": "beatsaver", "source_id": song_hash, "bpm": bpm}]
    (processed / "metadata.json").write_text(json.dumps(metadata))

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({song_hash: str(tmp_path / "dummy.wav")}))

    # Pre-populate mel cache so __init__ never needs real audio.
    cache_dir = processed / "mel_cache"
    cache_dir.mkdir()
    np.save(cache_dir / f"{song_hash}_{bpm}.npy", np.zeros((80, 100), dtype=np.float32))

    return processed, manifest_path


class TestIncludeBombsDisabled:
    def test_default_behavior_unaffected(self, tmp_path):
        """include_bombs=False (default) — bombs present in Parquet but never touched."""
        processed, manifest = _write_test_song(tmp_path, with_bombs=True)
        config = ModelConfig(max_seq_len=64, include_bombs=False)
        ds = BeatSaberDataset(processed, manifest, config, split="test")
        assert len(ds.samples) == 1
        tokens = ds.samples[0]["token_ids"]
        assert BOMB_EMPTY not in tokens
        assert not any(BOMB_BASE <= t < BOMB_BASE + BOMB_COUNT for t in tokens)


class TestIncludeBombsEnabled:
    def test_bomb_token_appears_in_sequence(self, tmp_path):
        processed, manifest = _write_test_song(tmp_path, with_bombs=True)
        config = ModelConfig(max_seq_len=64, include_bombs=True)
        ds = BeatSaberDataset(processed, manifest, config, split="test")
        assert len(ds.samples) == 1
        tokens = ds.samples[0]["token_ids"]
        bomb_tokens = [t for t in tokens if BOMB_BASE <= t < BOMB_BASE + BOMB_COUNT]
        assert len(bomb_tokens) == 1

    def test_no_bombs_parquet_at_all_does_not_crash(self, tmp_path):
        """include_bombs=True with zero bombs anywhere is a legitimate outcome, not an error."""
        processed, manifest = _write_test_song(tmp_path, with_bombs=False)
        config = ModelConfig(max_seq_len=64, include_bombs=True)
        ds = BeatSaberDataset(processed, manifest, config, split="test")
        assert len(ds.samples) == 1
        tokens = ds.samples[0]["token_ids"]
        # Every position should have BOMB_EMPTY since there are no bombs at all
        bomb_slots = [t for t in tokens if t == BOMB_EMPTY or (BOMB_BASE <= t < BOMB_BASE + BOMB_COUNT)]
        assert all(t == BOMB_EMPTY for t in bomb_slots)
        assert len(bomb_slots) > 0
