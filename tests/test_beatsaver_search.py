"""Tests for BeatSaverClient.search_maps' difficulty filter.

BeatSaver's search API has no server-side difficulty filter — this is
checked client-side against each map doc's own diffs list, mirroring how
min_score/min_upvotes are already filtered client-side per page.
"""
from unittest.mock import MagicMock, patch

from beat_weaver.sources.beatsaver import BeatSaverClient, _has_any_difficulty


def _doc(doc_id: str, difficulties: list[str], score: float = 0.9, upvotes: int = 10) -> dict:
    return {
        "id": doc_id,
        "automapper": False,
        "stats": {"score": score, "upvotes": upvotes},
        "versions": [
            {"diffs": [{"characteristic": "Standard", "difficulty": d} for d in difficulties]},
        ],
    }


class TestHasAnyDifficulty:
    def test_matching_difficulty_found(self):
        doc = _doc("a", ["Hard", "Expert", "ExpertPlus"])
        assert _has_any_difficulty(doc, {"Expert"})

    def test_no_matching_difficulty(self):
        doc = _doc("a", ["Easy", "Normal"])
        assert not _has_any_difficulty(doc, {"Expert", "ExpertPlus"})

    def test_empty_diffs_list(self):
        doc = {"versions": [{"diffs": []}]}
        assert not _has_any_difficulty(doc, {"Expert"})

    def test_no_versions_key(self):
        assert not _has_any_difficulty({}, {"Expert"})

    def test_checks_all_versions_not_just_first(self):
        doc = {
            "versions": [
                {"diffs": [{"characteristic": "Standard", "difficulty": "Easy"}]},
                {"diffs": [{"characteristic": "Standard", "difficulty": "ExpertPlus"}]},
            ],
        }
        assert _has_any_difficulty(doc, {"ExpertPlus"})


class TestSearchMapsDifficultyFilter:
    def _mock_response(self, docs: list[dict]):
        response = MagicMock()
        response.raise_for_status = MagicMock()
        response.json.return_value = {"docs": docs, "info": {"pages": 1}}
        return response

    def test_no_filter_yields_all(self):
        client = BeatSaverClient()
        docs = [_doc("a", ["Easy"]), _doc("b", ["Expert"])]
        with patch.object(client.session, "get", return_value=self._mock_response(docs)):
            results = list(client.search_maps(max_pages=1))
        assert {d["id"] for d in results} == {"a", "b"}

    def test_filter_keeps_only_matching_difficulty(self):
        client = BeatSaverClient()
        docs = [
            _doc("easy_only", ["Easy", "Normal"]),
            _doc("has_expert", ["Hard", "Expert"]),
            _doc("expert_plus_only", ["ExpertPlus"]),
        ]
        with patch.object(client.session, "get", return_value=self._mock_response(docs)):
            results = list(client.search_maps(max_pages=1, difficulties=["Expert"]))
        assert {d["id"] for d in results} == {"has_expert"}

    def test_filter_with_multiple_wanted_difficulties(self):
        client = BeatSaverClient()
        docs = [
            _doc("easy_only", ["Easy"]),
            _doc("normal_only", ["Normal"]),
            _doc("expert_only", ["Expert"]),
        ]
        with patch.object(client.session, "get", return_value=self._mock_response(docs)):
            results = list(
                client.search_maps(max_pages=1, difficulties=["Easy", "Normal"])
            )
        assert {d["id"] for d in results} == {"easy_only", "normal_only"}

    def test_score_and_difficulty_filters_combine(self):
        """A map must pass both the existing score filter and the new
        difficulty filter — neither one alone should be sufficient."""
        client = BeatSaverClient()
        docs = [
            _doc("low_score_right_diff", ["Expert"], score=0.5),
            _doc("high_score_wrong_diff", ["Easy"], score=0.95),
            _doc("high_score_right_diff", ["Expert"], score=0.95),
        ]
        with patch.object(client.session, "get", return_value=self._mock_response(docs)):
            results = list(
                client.search_maps(max_pages=1, min_score=0.75, difficulties=["Expert"])
            )
        assert {d["id"] for d in results} == {"high_score_right_diff"}
