"""Tests for cmd_generate's --audio/--audio-dir queue wiring.

The model/checkpoint loading and the actual per-song generation
(_generate_one) are mocked out — these tests only cover the queue
iteration logic: which files get picked up, what output path each gets,
and that one failing song doesn't stop the rest of the queue.
"""
import argparse
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

torch = pytest.importorskip("torch")

from beat_weaver.cli import cmd_generate


def _make_args(**overrides) -> argparse.Namespace:
    defaults = dict(
        checkpoint="fake_ckpt", audio=None, audio_dir=None,
        difficulty=["Expert"], output=None, bpm=None,
        temperature=1.0, seed=None, no_obstacles=False,
        obstacle_stats="data/processed/obstacle_stats.json",
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def _patched_cmd_generate(args, generate_one_mock):
    with patch("beat_weaver.model.config.ModelConfig.load") as mock_load_config, \
         patch("beat_weaver.model.transformer.BeatWeaverModel") as mock_model_cls, \
         patch("torch.load", return_value={}), \
         patch("beat_weaver.cli._generate_one", generate_one_mock):
        mock_load_config.return_value = MagicMock()
        mock_model_cls.return_value = MagicMock()
        cmd_generate(args)


class TestGenerateSingleFile:
    def test_single_audio_calls_generate_one_once(self, tmp_path):
        audio = tmp_path / "song.ogg"
        audio.write_bytes(b"fake")
        args = _make_args(audio=str(audio))
        mock_gen_one = MagicMock()

        _patched_cmd_generate(args, mock_gen_one)

        assert mock_gen_one.call_count == 1
        _, _, _, _, audio_path, output = mock_gen_one.call_args[0]
        assert audio_path == Path(audio)
        assert output == Path("output/song")

    def test_single_audio_respects_explicit_output(self, tmp_path):
        audio = tmp_path / "song.ogg"
        audio.write_bytes(b"fake")
        args = _make_args(audio=str(audio), output="my_map")
        mock_gen_one = MagicMock()

        _patched_cmd_generate(args, mock_gen_one)

        output = mock_gen_one.call_args[0][5]
        assert output == Path("my_map")


class TestGenerateAudioDirQueue:
    def test_processes_each_audio_file_once(self, tmp_path):
        for name in ("a.ogg", "b.wav", "c.mp3"):
            (tmp_path / name).write_bytes(b"fake")
        (tmp_path / "notes.txt").write_text("not audio")  # must be skipped

        args = _make_args(audio_dir=str(tmp_path), output="out_base")
        mock_gen_one = MagicMock()

        _patched_cmd_generate(args, mock_gen_one)

        assert mock_gen_one.call_count == 3
        processed_stems = sorted(call.args[4].stem for call in mock_gen_one.call_args_list)
        assert processed_stems == ["a", "b", "c"]

    def test_each_song_gets_its_own_output_subfolder(self, tmp_path):
        (tmp_path / "alpha.ogg").write_bytes(b"fake")
        args = _make_args(audio_dir=str(tmp_path), output="out_base")
        mock_gen_one = MagicMock()

        _patched_cmd_generate(args, mock_gen_one)

        output = mock_gen_one.call_args[0][5]
        assert output == Path("out_base") / "alpha"

    def test_default_output_base_when_not_specified(self, tmp_path):
        (tmp_path / "alpha.ogg").write_bytes(b"fake")
        args = _make_args(audio_dir=str(tmp_path), output=None)
        mock_gen_one = MagicMock()

        _patched_cmd_generate(args, mock_gen_one)

        output = mock_gen_one.call_args[0][5]
        assert output == Path("output") / "alpha"

    def test_empty_folder_does_not_crash(self, tmp_path):
        args = _make_args(audio_dir=str(tmp_path))
        mock_gen_one = MagicMock()

        _patched_cmd_generate(args, mock_gen_one)  # must not raise

        assert mock_gen_one.call_count == 0

    def test_one_failure_does_not_stop_the_queue(self, tmp_path):
        for name in ("a.ogg", "b.ogg", "c.ogg"):
            (tmp_path / name).write_bytes(b"fake")

        args = _make_args(audio_dir=str(tmp_path))
        mock_gen_one = MagicMock(side_effect=[None, RuntimeError("boom"), None])

        _patched_cmd_generate(args, mock_gen_one)  # must not raise/propagate

        assert mock_gen_one.call_count == 3
