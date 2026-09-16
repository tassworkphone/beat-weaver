"""Tests for training utilities — color balance loss."""

import pytest

torch = pytest.importorskip("torch")

from beat_weaver.model.config import ModelConfig
from beat_weaver.model.training import Trainer, _color_balance_loss
from beat_weaver.model.tokenizer import LEFT_BASE, LEFT_COUNT, RIGHT_BASE, RIGHT_COUNT
from beat_weaver.model.transformer import BeatWeaverModel


class TestTrainerWeightDecay:
    def test_config_weight_decay_reaches_optimizer(self, tmp_path):
        """A non-default weight_decay in config must actually reach AdamW —
        small-dataset training relies on this being tunable per-config."""
        config = ModelConfig(
            vocab_size=32, max_seq_len=16, n_mels=8, max_audio_len=16,
            encoder_layers=1, encoder_dim=8, encoder_heads=2, encoder_ff_dim=16,
            decoder_layers=1, decoder_dim=8, decoder_heads=2, decoder_ff_dim=16,
            weight_decay=0.05,
        )
        model = BeatWeaverModel(config)
        trainer = Trainer(model, config, tmp_path, device=torch.device("cpu"))
        assert trainer.optimizer.param_groups[0]["weight_decay"] == 0.05

    def test_default_weight_decay_unchanged(self, tmp_path):
        """Default weight_decay must match the previous hardcoded value (0.01)."""
        config = ModelConfig(
            vocab_size=32, max_seq_len=16, n_mels=8, max_audio_len=16,
            encoder_layers=1, encoder_dim=8, encoder_heads=2, encoder_ff_dim=16,
            decoder_layers=1, decoder_dim=8, decoder_heads=2, decoder_ff_dim=16,
        )
        model = BeatWeaverModel(config)
        trainer = Trainer(model, config, tmp_path, device=torch.device("cpu"))
        assert trainer.optimizer.param_groups[0]["weight_decay"] == 0.01


class TestColorBalanceLoss:
    def test_balanced_near_zero(self):
        """Loss near zero for 50/50 left/right predictions."""
        # Create logits that predict equal LEFT and RIGHT probability
        batch, seq, vocab = 2, 10, 291
        logits = torch.zeros(batch, seq, vocab)
        # Set equal logits for all LEFT and RIGHT tokens
        logits[:, :, LEFT_BASE:LEFT_BASE + LEFT_COUNT] = 1.0
        logits[:, :, RIGHT_BASE:RIGHT_BASE + RIGHT_COUNT] = 1.0
        loss = _color_balance_loss(logits)
        assert loss.item() < 0.01

    def test_imbalanced_positive(self):
        """Loss > 0 for 100% left predictions."""
        batch, seq, vocab = 2, 10, 291
        logits = torch.full((batch, seq, vocab), -10.0)
        # All probability on LEFT tokens
        logits[:, :, LEFT_BASE:LEFT_BASE + LEFT_COUNT] = 5.0
        loss = _color_balance_loss(logits)
        assert loss.item() > 0.1

    def test_gradient_flows(self):
        """Gradients flow through auxiliary loss."""
        batch, seq, vocab = 2, 10, 291
        logits = torch.randn(batch, seq, vocab, requires_grad=True)
        loss = _color_balance_loss(logits)
        loss.backward()
        assert logits.grad is not None
        assert logits.grad.abs().sum() > 0

    def test_no_note_tokens(self):
        """Returns 0 when no positions have significant note probability."""
        batch, seq, vocab = 2, 10, 291
        logits = torch.zeros(batch, seq, vocab)
        # All probability on PAD/START/END tokens, none on notes
        logits[:, :, 0:8] = 10.0
        loss = _color_balance_loss(logits)
        assert loss.item() == 0.0
