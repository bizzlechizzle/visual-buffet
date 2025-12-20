"""Tests for hardware detection."""

from unittest.mock import MagicMock, patch

import pytest

from imlage.core.hardware import (
    CACHE_DIR,
    CACHE_FILE,
    _detect_gpu,
    detect_hardware,
    get_recommended_batch_size,
)
from imlage.plugins.schemas import HardwareProfile


class TestGPUDetection:
    """Tests for GPU detection."""

    def test_detect_gpu_no_torch(self):
        """Test GPU detection without torch installed."""
        with patch.dict("sys.modules", {"torch": None}):
            gpu_type, gpu_name, vram = _detect_gpu()

        # Should gracefully return None values
        assert gpu_type is None
        assert gpu_name is None
        assert vram is None

    def test_detect_gpu_cuda(self):
        """Test CUDA GPU detection."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.get_device_name.return_value = "NVIDIA RTX 3080"
        mock_props = MagicMock()
        mock_props.total_memory = 10 * 1024**3  # 10 GB
        mock_torch.cuda.get_device_properties.return_value = mock_props
        mock_torch.backends.mps.is_available.return_value = False

        with patch.dict("sys.modules", {"torch": mock_torch}):
            gpu_type, gpu_name, vram = _detect_gpu()

        assert gpu_type == "cuda"
        assert gpu_name == "NVIDIA RTX 3080"
        assert vram == pytest.approx(10.0, rel=0.1)

    def test_detect_gpu_mps(self):
        """Test MPS (Apple Silicon) GPU detection."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = True

        with patch.dict("sys.modules", {"torch": mock_torch}):
            gpu_type, gpu_name, vram = _detect_gpu()

        assert gpu_type == "mps"
        assert gpu_name == "Apple Silicon"
        assert vram is None

    def test_detect_gpu_none(self):
        """Test no GPU detected."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False

        with patch.dict("sys.modules", {"torch": mock_torch}):
            gpu_type, gpu_name, vram = _detect_gpu()

        assert gpu_type is None
        assert gpu_name is None
        assert vram is None


class TestDetectHardware:
    """Tests for main hardware detection function."""

    @patch("imlage.core.hardware._detect_gpu")
    @patch("imlage.core.hardware.psutil")
    @patch("imlage.core.hardware.platform")
    @patch("imlage.core.hardware._save_cache")
    @patch("imlage.core.hardware.CACHE_FILE")
    def test_detect_hardware_fresh(
        self, mock_cache_file, mock_save, mock_platform, mock_psutil, mock_gpu
    ):
        """Test fresh hardware detection."""
        mock_cache_file.exists.return_value = False
        mock_platform.processor.return_value = "Intel i7"
        mock_psutil.cpu_count.return_value = 8

        mock_mem = MagicMock()
        mock_mem.total = 16 * 1024**3
        mock_mem.available = 12 * 1024**3
        mock_psutil.virtual_memory.return_value = mock_mem

        mock_gpu.return_value = ("cuda", "RTX 3080", 10.0)

        profile = detect_hardware(force_refresh=True)

        assert profile.cpu_model == "Intel i7"
        assert profile.cpu_cores == 8
        assert profile.ram_total_gb == 16.0
        assert profile.gpu_type == "cuda"

    @patch("imlage.core.hardware._load_cached")
    @patch("imlage.core.hardware.CACHE_FILE")
    def test_detect_hardware_cached(self, mock_cache_file, mock_load):
        """Test hardware detection from cache."""
        mock_cache_file.exists.return_value = True
        cached_profile = HardwareProfile(
            cpu_model="Cached CPU",
            cpu_cores=4,
            ram_total_gb=8.0,
            ram_available_gb=4.0,
        )
        mock_load.return_value = cached_profile

        profile = detect_hardware(force_refresh=False)

        assert profile.cpu_model == "Cached CPU"


class TestBatchSizeRecommendation:
    """Tests for batch size recommendations."""

    def test_batch_size_low_ram_cpu_only(self):
        """Test batch size for low RAM, CPU only."""
        profile = HardwareProfile(
            cpu_model="Test",
            cpu_cores=4,
            ram_total_gb=6.0,
            ram_available_gb=4.0,
        )

        batch_size = get_recommended_batch_size(profile)
        assert batch_size == 1

    def test_batch_size_high_ram_cpu_only(self):
        """Test batch size for high RAM, CPU only."""
        profile = HardwareProfile(
            cpu_model="Test",
            cpu_cores=8,
            ram_total_gb=32.0,
            ram_available_gb=24.0,
        )

        batch_size = get_recommended_batch_size(profile)
        assert batch_size == 2

    def test_batch_size_cuda_low_vram(self):
        """Test batch size for CUDA with low VRAM."""
        profile = HardwareProfile(
            cpu_model="Test",
            cpu_cores=8,
            ram_total_gb=16.0,
            ram_available_gb=12.0,
            gpu_type="cuda",
            gpu_name="GTX 1060",
            gpu_vram_gb=4.0,
        )

        batch_size = get_recommended_batch_size(profile)
        assert batch_size == 4

    def test_batch_size_cuda_high_vram(self):
        """Test batch size for CUDA with high VRAM."""
        profile = HardwareProfile(
            cpu_model="Test",
            cpu_cores=8,
            ram_total_gb=32.0,
            ram_available_gb=24.0,
            gpu_type="cuda",
            gpu_name="RTX 3080",
            gpu_vram_gb=10.0,
        )

        batch_size = get_recommended_batch_size(profile)
        assert batch_size == 8

    def test_batch_size_mps(self):
        """Test batch size for Apple Silicon."""
        profile = HardwareProfile(
            cpu_model="Apple M2 Pro",
            cpu_cores=12,
            ram_total_gb=32.0,
            ram_available_gb=24.0,
            gpu_type="mps",
            gpu_name="Apple M2 Pro",
        )

        batch_size = get_recommended_batch_size(profile)
        assert batch_size == 4


class TestCacheConstants:
    """Tests for cache constants."""

    def test_cache_dir_in_home(self):
        """Test cache dir is in home directory."""
        assert ".imlage" in str(CACHE_DIR)

    def test_cache_file_name(self):
        """Test cache file has correct name."""
        assert CACHE_FILE.name == "hardware.json"
