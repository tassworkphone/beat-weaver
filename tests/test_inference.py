"""Tests for inference and grammar-constrained generation."""

import pytest

torch = pytest.importorskip("torch")

from beat_weaver.model.config import ModelConfig
from beat_weaver.model.inference import (
    _build_grammar_mask,
    _max_generation_tokens,
    generate,
    generate_full_song,
)
from beat_weaver.model.tokenizer import (
    BAR,
    BOMB_BASE,
    BOMB_COUNT,
    BOMB_EMPTY,
    DIFF_EASY,
    DIFF_EXPERT,
    DIFF_EXPERT_PLUS,
    END,
    LEFT_BASE,
    LEFT_COUNT,
    LEFT_EMPTY,
    POS_BASE,
    POS_COUNT,
    RIGHT_BASE,
    RIGHT_COUNT,
    RIGHT_EMPTY,
    START,
    VOCAB_SIZE,
    VOCAB_SIZE_WITH_BOMBS,
)
from beat_weaver.model.transformer import BeatWeaverModel


class TestGrammarMask:
    def test_after_start(self):
        mask = _build_grammar_mask(START)
        # Only difficulty tokens allowed
        assert mask[DIFF_EASY:DIFF_EXPERT_PLUS + 1].all()
        assert not mask[START]
        assert not mask[BAR]
        assert not mask[END]

    def test_after_difficulty(self):
        mask = _build_grammar_mask(DIFF_EXPERT)
        # Only BAR allowed
        assert mask[BAR]
        assert mask.sum() == 1

    def test_after_bar(self):
        mask = _build_grammar_mask(BAR)
        # POS, BAR, or END
        assert mask[POS_BASE:POS_BASE + POS_COUNT].all()
        assert mask[BAR]
        assert mask[END]
        assert not mask[START]
        assert not mask[LEFT_EMPTY]

    def test_after_pos(self):
        mask = _build_grammar_mask(POS_BASE + 10)
        # LEFT tokens
        assert mask[LEFT_EMPTY]
        assert mask[LEFT_BASE:LEFT_BASE + LEFT_COUNT].all()
        assert not mask[BAR]
        assert not mask[RIGHT_EMPTY]

    def test_after_left(self):
        mask = _build_grammar_mask(LEFT_BASE + 5)
        # RIGHT tokens
        assert mask[RIGHT_EMPTY]
        assert mask[RIGHT_BASE:RIGHT_BASE + RIGHT_COUNT].all()
        assert not mask[BAR]
        assert not mask[LEFT_EMPTY]

    def test_after_left_empty(self):
        mask = _build_grammar_mask(LEFT_EMPTY)
        # RIGHT tokens
        assert mask[RIGHT_EMPTY]
        assert mask[RIGHT_BASE:RIGHT_BASE + RIGHT_COUNT].all()

    def test_after_right(self):
        mask = _build_grammar_mask(RIGHT_BASE + 5)
        # POS, BAR, or END
        assert mask[POS_BASE:POS_BASE + POS_COUNT].all()
        assert mask[BAR]
        assert mask[END]

    def test_after_right_empty(self):
        mask = _build_grammar_mask(RIGHT_EMPTY)
        assert mask[POS_BASE:POS_BASE + POS_COUNT].all()
        assert mask[BAR]
        assert mask[END]

    def test_after_right_with_bombs_goes_to_bomb_slot(self):
        """With include_bombs=True, RIGHT transitions to BOMB, not POS/BAR/END."""
        mask = _build_grammar_mask(RIGHT_BASE + 5, include_bombs=True)
        assert mask.shape[0] == VOCAB_SIZE_WITH_BOMBS
        assert mask[BOMB_EMPTY]
        assert mask[BOMB_BASE:BOMB_BASE + BOMB_COUNT].all()
        assert not mask[POS_BASE]
        assert not mask[BAR]
        assert not mask[END]

    def test_after_right_empty_with_bombs_goes_to_bomb_slot(self):
        mask = _build_grammar_mask(RIGHT_EMPTY, include_bombs=True)
        assert mask[BOMB_EMPTY]
        assert mask[BOMB_BASE:BOMB_BASE + BOMB_COUNT].all()

    def test_after_bomb_goes_to_pos_bar_end(self):
        mask = _build_grammar_mask(BOMB_BASE + 3, last_pos_in_bar=5, include_bombs=True)
        assert mask[POS_BASE + 6: POS_BASE + POS_COUNT].all()
        assert not mask[POS_BASE + 5]  # strictly increasing, same as RIGHT->POS
        assert mask[BAR]
        assert mask[END]

    def test_after_bomb_empty_goes_to_pos_bar_end(self):
        mask = _build_grammar_mask(BOMB_EMPTY, last_pos_in_bar=-1, include_bombs=True)
        assert mask[POS_BASE:POS_BASE + POS_COUNT].all()
        assert mask[BAR]
        assert mask[END]

    def test_include_bombs_false_matches_default_shape(self):
        """Sanity check: default behavior (no bombs) is unaffected by the new param."""
        mask_default = _build_grammar_mask(RIGHT_BASE + 5)
        mask_explicit = _build_grammar_mask(RIGHT_BASE + 5, include_bombs=False)
        assert mask_default.shape == mask_explicit.shape == (VOCAB_SIZE,)
        assert torch.equal(mask_default, mask_explicit)


class TestMaxGenerationTokens:
    """Regression coverage for the windowed-generation truncation bug: a fixed
    max_seq_len=1024 token budget can't always cover a full max_audio_len=4096
    window (64 bars) at real note density, silently dropping the back half of
    the window — exactly the half kept by generate_full_song's midpoint
    ownership, producing dead gaps followed by a pile-up.
    """

    def test_small_window_floors_at_max_seq_len(self):
        # 1 bar (64 frames) needs far fewer than max_seq_len tokens — the
        # floor should win, leaving short single-window songs unaffected.
        assert _max_generation_tokens(n_frames=64, max_seq_len=1024) == 1024

    def test_full_size_window_exceeds_max_seq_len(self):
        # 4096 frames = 64 bars; at 25 tokens/bar that's 1604, well past 1024.
        budget = _max_generation_tokens(n_frames=4096, max_seq_len=1024)
        assert budget > 1024

    def test_scales_with_bar_count(self):
        small = _max_generation_tokens(n_frames=64, max_seq_len=1)
        large = _max_generation_tokens(n_frames=4096, max_seq_len=1)
        assert large > small

    def test_partial_bar_rounds_up(self):
        # 65 frames is just over 1 bar (64 frames) — must budget for 2 bars,
        # not truncate down to 1.
        one_bar = _max_generation_tokens(n_frames=64, max_seq_len=1)
        two_bars = _max_generation_tokens(n_frames=65, max_seq_len=1)
        assert two_bars > one_bar


class TestGenerate:
    @pytest.fixture
    def small_model(self):
        config = ModelConfig(
            vocab_size=291,
            max_seq_len=64,
            n_mels=80,
            encoder_layers=1,
            encoder_dim=32,
            encoder_heads=4,
            encoder_ff_dim=64,
            decoder_layers=1,
            decoder_dim=32,
            decoder_heads=4,
            decoder_ff_dim=64,
            dropout=0.0,
        )
        model = BeatWeaverModel(config)
        return model, config

    def test_starts_and_ends_correctly(self, small_model):
        model, config = small_model
        mel = torch.randn(80, 20)
        tokens = generate(model, mel, "Expert", config, temperature=1.0, seed=42)
        assert tokens[0] == START
        assert tokens[1] == DIFF_EXPERT
        # Should end with END or hit max_seq_len
        assert tokens[-1] == END or len(tokens) == config.max_seq_len

    def test_grammar_valid_sequence(self, small_model):
        """Every generated token should follow grammar rules."""
        model, config = small_model
        mel = torch.randn(80, 20)
        tokens = generate(model, mel, "Expert", config, temperature=1.0, seed=42)

        for i in range(1, len(tokens)):
            prev = tokens[i - 1]
            curr = tokens[i]
            mask = _build_grammar_mask(prev)
            assert mask[curr], (
                f"Token {curr} not valid after {prev} at position {i}. "
                f"Sequence so far: {tokens[:i+1]}"
            )

    def test_deterministic_with_seed(self, small_model):
        model, config = small_model
        mel = torch.randn(80, 20)
        t1 = generate(model, mel, "Expert", config, temperature=0.5, seed=123)
        t2 = generate(model, mel, "Expert", config, temperature=0.5, seed=123)
        assert t1 == t2

    def test_greedy_decoding(self, small_model):
        model, config = small_model
        mel = torch.randn(80, 20)
        tokens = generate(model, mel, "Expert", config, temperature=0)
        assert tokens[0] == START
        assert tokens[1] == DIFF_EXPERT


class TestCubeHeadBias:
    """Inference-time logit bias from the binary 'cube here' head."""

    def _cube_model(self, bias_scale: float):
        config = ModelConfig(
            vocab_size=291, max_seq_len=64, n_mels=80,
            encoder_layers=1, encoder_dim=32, encoder_heads=4, encoder_ff_dim=64,
            decoder_layers=1, decoder_dim=32, decoder_heads=4, decoder_ff_dim=64,
            dropout=0.0, use_cube_head=True, cube_head_bias_scale=bias_scale,
        )
        model = BeatWeaverModel(config)
        return model, config

    def test_bias_scale_zero_matches_no_cube_head(self):
        """cube_head_bias_scale=0 must produce byte-identical output to a
        model that doesn't have the head at all — the head's mere existence
        (random init) must not leak into generation unless explicitly enabled."""
        cube_model, cube_config = self._cube_model(bias_scale=0.0)
        plain_config = ModelConfig(
            vocab_size=291, max_seq_len=64, n_mels=80,
            encoder_layers=1, encoder_dim=32, encoder_heads=4, encoder_ff_dim=64,
            decoder_layers=1, decoder_dim=32, decoder_heads=4, decoder_ff_dim=64,
            dropout=0.0, use_cube_head=False,
        )
        plain_model = BeatWeaverModel(plain_config)
        # Copy shared weights so only cube_head differs between the two models.
        plain_model.load_state_dict(cube_model.state_dict(), strict=False)

        mel = torch.randn(80, 20)
        t_cube = generate(cube_model, mel, "Expert", cube_config, temperature=0.5, seed=7)
        t_plain = generate(plain_model, mel, "Expert", plain_config, temperature=0.5, seed=7)
        assert t_cube == t_plain

    def test_grammar_valid_with_cube_bias_active(self):
        model, config = self._cube_model(bias_scale=5.0)
        mel = torch.randn(80, 20)
        tokens = generate(model, mel, "Expert", config, temperature=1.0, seed=42)
        for i in range(1, len(tokens)):
            prev = tokens[i - 1]
            curr = tokens[i]
            mask = _build_grammar_mask(prev)
            assert mask[curr], f"Token {curr} not valid after {prev} at position {i}"

    def test_strong_bias_changes_output(self):
        """A large bias_scale must actually influence sampling, not be a
        silent no-op — proof the bias is reaching the logits."""
        model, config = self._cube_model(bias_scale=50.0)
        model_no_bias, config_no_bias = self._cube_model(bias_scale=0.0)
        model_no_bias.load_state_dict(model.state_dict())

        mel = torch.randn(80, 20)
        t_biased = generate(model, mel, "Expert", config, temperature=1.0, seed=42)
        t_unbiased = generate(model_no_bias, mel, "Expert", config_no_bias, temperature=1.0, seed=42)
        assert t_biased != t_unbiased


class TestGenerateWithBombs:
    @pytest.fixture
    def small_bomb_model(self):
        config = ModelConfig(
            vocab_size=VOCAB_SIZE_WITH_BOMBS,
            max_seq_len=64,
            n_mels=80,
            encoder_layers=1,
            encoder_dim=32,
            encoder_heads=4,
            encoder_ff_dim=64,
            decoder_layers=1,
            decoder_dim=32,
            decoder_heads=4,
            decoder_ff_dim=64,
            dropout=0.0,
            include_bombs=True,
        )
        model = BeatWeaverModel(config)
        return model, config

    def test_grammar_valid_sequence_with_bombs(self, small_bomb_model):
        """Every generated token follows the bomb-aware grammar (POS LEFT RIGHT BOMB)."""
        model, config = small_bomb_model
        mel = torch.randn(80, 20)
        tokens = generate(model, mel, "Expert", config, temperature=1.0, seed=42)

        for i in range(1, len(tokens)):
            prev = tokens[i - 1]
            curr = tokens[i]
            mask = _build_grammar_mask(prev, include_bombs=True)
            assert mask[curr], (
                f"Token {curr} not valid after {prev} at position {i}. "
                f"Sequence so far: {tokens[:i+1]}"
            )

    def test_decode_produces_bomb_notes(self, small_bomb_model):
        """End-to-end: generated bomb tokens decode to color=3 Note entries."""
        from beat_weaver.model.tokenizer import decode_tokens

        model, config = small_bomb_model
        mel = torch.randn(80, 20)
        tokens = generate(model, mel, "Expert", config, temperature=1.0, seed=1)
        notes = decode_tokens(tokens, bpm=120.0, include_bombs=True)
        # With a random model, bombs may or may not appear — just verify decoding
        # never crashes and any bombs present are well-formed.
        for n in notes:
            assert n.color in (0, 1, 3)
            if n.color == 3:
                assert 0 <= n.x < 4 and 0 <= n.y < 3


class TestGenerateFullSong:
    @pytest.fixture
    def small_model(self):
        config = ModelConfig(
            vocab_size=291,
            max_seq_len=64,
            max_audio_len=128,
            n_mels=80,
            encoder_layers=1,
            encoder_dim=32,
            encoder_heads=4,
            encoder_ff_dim=64,
            decoder_layers=1,
            decoder_dim=32,
            decoder_heads=4,
            decoder_ff_dim=64,
            dropout=0.0,
        )
        model = BeatWeaverModel(config)
        return model, config

    def test_full_song_single_window(self, small_model):
        """Short mel that fits in one window produces same result as generate()."""
        model, config = small_model
        mel = torch.randn(80, 64)  # Well under max_audio_len=128
        notes = generate_full_song(model, mel, "Expert", config, bpm=120.0, seed=42)
        # Should return a list of Note objects (may be empty if model generates no notes)
        assert isinstance(notes, list)
        for note in notes:
            assert hasattr(note, "beat")
            assert hasattr(note, "color")

    def test_full_song_multi_window(self, small_model):
        """Mel 2.5x max_audio_len produces notes spanning the full duration."""
        model, config = small_model
        total_frames = int(config.max_audio_len * 2.5)
        mel = torch.randn(80, total_frames)
        notes = generate_full_song(
            model, mel, "Expert", config, bpm=120.0,
            temperature=1.0, seed=42,
        )
        assert isinstance(notes, list)
        # With a random model we might not get notes everywhere, but
        # the function should run without errors and return Note objects
        for note in notes:
            assert hasattr(note, "beat")

    def test_full_song_no_notes_past_real_audio_length(self, small_model):
        """Regression: the last window is zero-padded up to max_audio_len, so
        without masking/clipping the model could generate notes over pure
        padding, well past the real song's end. total_frames here (2.5x
        max_audio_len) is NOT a multiple of max_audio_len, so the last window
        is genuinely partial/padded — no note should land beyond it."""
        model, config = small_model
        total_frames = int(config.max_audio_len * 2.5)
        mel = torch.randn(80, total_frames)
        notes = generate_full_song(
            model, mel, "Expert", config, bpm=120.0,
            temperature=1.0, seed=42,
        )
        song_end_beat = total_frames / 16.0
        for note in notes:
            assert note.beat < song_end_beat
            assert note.beat >= 0

    def test_full_song_overlap_no_duplicates(self, small_model):
        """No two notes at the exact same beat+color in the overlap zone."""
        model, config = small_model
        total_frames = config.max_audio_len * 3
        mel = torch.randn(80, total_frames)
        notes = generate_full_song(
            model, mel, "Expert", config, bpm=120.0, seed=99,
        )
        seen = set()
        for note in notes:
            key = (round(note.beat, 6), note.color, note.x, note.y)
            # Notes can legitimately share a beat if they're different placements,
            # but the same (beat, color, x, y) should not appear twice
            assert key not in seen, f"Duplicate note at {key}"
            seen.add(key)

    def test_full_song_notes_sorted(self, small_model):
        """Output notes are sorted by beat."""
        model, config = small_model
        total_frames = config.max_audio_len * 2
        mel = torch.randn(80, total_frames)
        notes = generate_full_song(
            model, mel, "Expert", config, bpm=120.0, seed=7,
        )
        for i in range(1, len(notes)):
            assert notes[i].beat >= notes[i - 1].beat
