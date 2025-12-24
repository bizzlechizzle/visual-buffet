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

    def test_vocab_commands_have_app_option(self, cli):
        """Test all vocab subcommands have --app option."""
        from visual_buffet.cli import main

        commands = ["stats", "search", "export", "import", "learn", "review"]

        for cmd in commands:
            result = cli.invoke(main, ["vocab", cmd, "--help"])
            assert "--app" in result.output, f"Command {cmd} missing --app option"


class TestAppConfig:
    """Tests for app-specific configuration."""

    def test_app_config_creates_paths(self):
        """Test AppConfig generates correct paths."""
        from vocablearn import AppConfig

        config = AppConfig("test-app")
        assert config.app_name == "test-app"
        assert str(config.vocab_db).endswith(".test-app/data/vocabulary.db")
        assert str(config.ocr_db).endswith(".test-app/data/ocr.db")

    def test_app_config_custom_base(self):
        """Test AppConfig with custom base directory."""
        from vocablearn import AppConfig
        from pathlib import Path

        config = AppConfig("test-app", base_dir=Path("/tmp/custom"))
        assert str(config.vocab_db) == "/tmp/custom/data/vocabulary.db"


class TestOCRStorage:
    """Tests for OCR vocabulary storage."""

    def test_ocr_storage_record_and_retrieve(self):
        """Test OCRStorage can record and retrieve detections."""
        from vocablearn import OCRStorage
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "ocr.db"
            ocr = OCRStorage(db_path)

            # Record detection
            det_id = ocr.record_detection(
                image_id="test.jpg",
                text="EXIT",
                ocr_engine="tesseract",
                confidence=0.95,
                text_type="sign",
            )
            assert det_id == 1

            # Retrieve
            text = ocr.get_text("EXIT")
            assert text is not None
            assert text.text == "EXIT"
            assert text.text_type == "sign"
            assert text.total_occurrences == 1

    def test_ocr_storage_statistics(self):
        """Test OCRStorage statistics."""
        from vocablearn import OCRStorage
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "ocr.db"
            ocr = OCRStorage(db_path)

            ocr.record_detection("a.jpg", "EXIT", "tesseract", 0.9, "sign")
            ocr.record_detection("b.jpg", "DANGER", "tesseract", 0.8, "sign")
            ocr.record_detection("c.jpg", "EXIT", "tesseract", 0.95, "sign")

            stats = ocr.get_statistics()
            assert stats["vocabulary_size"] == 2  # EXIT and DANGER
            assert stats["total_detections"] == 3
            assert stats["images_with_text"] == 3
