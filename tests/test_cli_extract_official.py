"""Tests for cmd_extract_official's argument wiring.

Regression coverage for a bug where cmd_extract_official built `bundles_dir`
from `args.beat_saber` correctly, but never passed `beat_saber_path` through
to extract_official_maps() — which silently fell back to its hardcoded
default Steam path for finding level bundles, so a non-default --beat-saber
install location (e.g. a D: drive install) always found zero levels.
"""
import argparse
from pathlib import Path
from unittest.mock import patch

from beat_weaver.cli import cmd_extract_official


class TestExtractOfficialArgWiring:
    def test_beat_saber_path_passed_through(self):
        custom_path = r"D:\Beat Saber"
        args = argparse.Namespace(beat_saber=custom_path, output="data/raw/official")

        with patch("beat_weaver.sources.unity_extractor.extract_official_maps") as mock_extract:
            mock_extract.return_value = []
            cmd_extract_official(args)

        assert mock_extract.called
        _, kwargs = mock_extract.call_args
        assert kwargs["beat_saber_path"] == Path(custom_path)

    def test_bundles_dir_derived_from_custom_path(self):
        custom_path = r"D:\Beat Saber"
        args = argparse.Namespace(beat_saber=custom_path, output="data/raw/official")

        with patch("beat_weaver.sources.unity_extractor.extract_official_maps") as mock_extract:
            mock_extract.return_value = []
            cmd_extract_official(args)

        call_args, _ = mock_extract.call_args
        bundles_dir = call_args[0]
        assert bundles_dir == Path(custom_path) / "Beat Saber_Data" / "StreamingAssets" / "aa" / "StandaloneWindows64"
