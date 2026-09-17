"""Regression test for cmd_evaluate's reference-notes reconstruction.

Bug: BeatSaberDataset.__init__ frees each sample's raw `notes` dicts after
tokenization (a memory optimization for training — only token_ids/token_mask
are needed there), but cmd_evaluate read sample["notes"] directly to build
reference Note objects for comparison, causing a KeyError on every call.
Fix: reconstruct reference notes from sample["token_ids"] via decode_tokens()
instead, since that's the data BeatSaberDataset actually keeps.
"""
import argparse
import json

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

from beat_weaver.cli import cmd_evaluate
from beat_weaver.model.config import ModelConfig
from beat_weaver.model.transformer import BeatWeaverModel


def _write_test_song(tmp_path, song_hash="hash_0001", bpm=120.0):
    from beat_weaver.storage.writer import NOTES_SCHEMA

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

    metadata = [{"hash": song_hash, "source": "beatsaver", "source_id": song_hash, "bpm": bpm}]
    (processed / "metadata.json").write_text(json.dumps(metadata))

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({song_hash: str(tmp_path / "dummy.wav")}))

    cache_dir = processed / "mel_cache"
    cache_dir.mkdir()
    np.save(cache_dir / f"{song_hash}_{bpm}.npy", np.zeros((80, 100), dtype=np.float32))

    return processed, manifest_path


def _save_tiny_checkpoint(ckpt_dir):
    config = ModelConfig(
        max_seq_len=64, n_mels=80,
        encoder_layers=1, encoder_dim=32, encoder_heads=4, encoder_ff_dim=64,
        decoder_layers=1, decoder_dim=32, decoder_heads=4, decoder_ff_dim=64,
        dropout=0.0,
    )
    model = BeatWeaverModel(config)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    config.save(ckpt_dir / "config.json")
    torch.save(model.state_dict(), ckpt_dir / "model.pt")
    return config


class TestCmdEvaluate:
    def test_runs_without_keyerror_on_notes(self, tmp_path):
        """cmd_evaluate must not crash reconstructing reference notes — this
        reproduces the exact single-song 'test' split scenario that triggered
        the KeyError('notes') bug."""
        processed, manifest = _write_test_song(tmp_path)
        ckpt_dir = tmp_path / "checkpoint"
        _save_tiny_checkpoint(ckpt_dir)

        output_path = tmp_path / "eval_results.json"
        args = argparse.Namespace(
            checkpoint=str(ckpt_dir),
            data=str(processed),
            audio_manifest=str(manifest),
            output=str(output_path),
        )

        cmd_evaluate(args)  # must not raise KeyError

        results = json.loads(output_path.read_text())
        assert len(results) == 1
        expected_keys = {
            "onset_f1", "nps_accuracy", "beat_alignment",
            "parity_violation_rate", "pattern_diversity", "nps",
            "song_hash", "difficulty",
        }
        assert expected_keys.issubset(results[0].keys())

    def test_teacher_forced_writes_class_accuracy(self, tmp_path):
        processed, manifest = _write_test_song(tmp_path)
        ckpt_dir = tmp_path / "checkpoint"
        _save_tiny_checkpoint(ckpt_dir)

        output_path = tmp_path / "eval_tf.json"
        args = argparse.Namespace(
            checkpoint=str(ckpt_dir),
            data=str(processed),
            audio_manifest=str(manifest),
            output=str(output_path),
            split="test",
            mode="teacher-forced",
            max_maps=None,
        )

        cmd_evaluate(args)

        report = json.loads(output_path.read_text())
        assert "teacher_forced" in report
        tf = report["teacher_forced"]
        for key in ("overall", "cube", "cube_empty", "bomb", "bomb_empty", "structure"):
            assert key in tf
            assert "correct" in tf[key]
            assert "total" in tf[key]
        assert tf["overall"]["total"] > 0
