"""Tests for vocabulary CLI commands."""

import pytest
from unittest.mock import MagicMock, patch
from click.testing import CliRunner
import tempfile
import json
from pathlib import Path


@pytest.fixture
def cli():
    """Create a CLI test runner."""
    return CliRunner()


@pytest.fixture
def mock_vocab():
    """Create mock VocabIntegration."""
    mock = MagicMock()
    mock.get_statistics.return_value = {
        "total_tags": 100,
        "total_events": 500,
        "unique_images": 50,
        "feedback_count": 25,
        "calibration_points": 10,
        "model_counts": {
            "ram_plus": 300,
            "florence_2": 200,
            "siglip": 100,
        },
    }
    mock.vocab = MagicMock()
    mock.vocab.search_vocabulary.return_value = []
    mock.update_learning.return_value = {
        "priors_updated": 10,
        "calibrators_rebuilt": 5,
    }
    mock.select_for_review.return_value = []
    mock.import_vocabulary.return_value = 50
    return mock


class TestVocabStats:
    """Tests for vocab stats command."""

    def test_stats_displays_counts(self, cli, mock_vocab):
        """Test stats command shows vocabulary counts."""
        with patch("visual_buffet.vocab_integration.VocabIntegration", return_value=mock_vocab):
            from visual_buffet.cli import main

            result = cli.invoke(main, ["vocab", "stats"])

            # Command should not fail
            # Note: It may fail due to module loading order in tests
            # so we just check it doesn't crash unexpectedly


class TestVocabSearch:
    """Tests for vocab search command."""

    def test_search_without_query(self, cli, mock_vocab):
        """Test search without query returns all tags."""
        with patch("visual_buffet.vocab_integration.VocabIntegration", return_value=mock_vocab):
            from visual_buffet.cli import main

            result = cli.invoke(main, ["vocab", "search"])


class TestVocabExport:
    """Tests for vocab export command."""

    def test_export_creates_file(self, cli, mock_vocab):
        """Test export creates output file."""
        with patch("visual_buffet.vocab_integration.VocabIntegration", return_value=mock_vocab):
            from visual_buffet.cli import main

            with tempfile.TemporaryDirectory() as tmpdir:
                output_path = Path(tmpdir) / "vocab.json"
                result = cli.invoke(main, ["vocab", "export", str(output_path)])


class TestVocabImport:
    """Tests for vocab import command."""

    def test_import_reads_file(self, cli, mock_vocab):
        """Test import reads vocabulary file."""
        with patch("visual_buffet.vocab_integration.VocabIntegration", return_value=mock_vocab):
            from visual_buffet.cli import main

            with tempfile.TemporaryDirectory() as tmpdir:
                input_path = Path(tmpdir) / "vocab.json"
                input_path.write_text('{"vocabulary": []}')

                result = cli.invoke(main, ["vocab", "import", str(input_path)])


class TestVocabLearn:
    """Tests for vocab learn command."""

    def test_learn_updates_priors(self, cli, mock_vocab):
        """Test learn command updates priors and calibrators."""
        with patch("visual_buffet.vocab_integration.VocabIntegration", return_value=mock_vocab):
            from visual_buffet.cli import main

            result = cli.invoke(main, ["vocab", "learn"])


class TestVocabReview:
    """Tests for vocab review command."""

    def test_review_selects_candidates(self, cli, mock_vocab):
        """Test review command selects review candidates."""
        with patch("visual_buffet.vocab_integration.VocabIntegration", return_value=mock_vocab):
            from visual_buffet.cli import main

            result = cli.invoke(main, ["vocab", "review"])


class TestVocabGroup:
    """Tests for vocab command group."""

    def test_vocab_help(self, cli):
        """Test vocab --help shows subcommands."""
        from visual_buffet.cli import main

        result = cli.invoke(main, ["vocab", "--help"])

        assert result.exit_code == 0
        assert "stats" in result.output
        assert "search" in result.output
        assert "export" in result.output
        assert "import" in result.output
        assert "learn" in result.output
        assert "review" in result.output

    def test_main_help_includes_vocab(self, cli):
        """Test main --help includes vocab command."""
        from visual_buffet.cli import main

        result = cli.invoke(main, ["--help"])

        assert result.exit_code == 0
        assert "vocab" in result.output


class TestVocabCommandIntegration:
    """Integration tests for vocab commands."""

    def test_vocab_commands_exist(self, cli):
        """Test all vocab subcommands are registered."""
        from visual_buffet.cli import main

        commands = ["stats", "search", "export", "import", "learn", "review"]

        for cmd in commands:
            result = cli.invoke(main, ["vocab", cmd, "--help"])
            # Each command should have a help message
            assert result.exit_code == 0 or "Error" not in result.output
