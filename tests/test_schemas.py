"""Tests for plugin schemas."""


from imlage.plugins.schemas import HardwareProfile, PluginInfo, Tag, TagResult


class TestTag:
    """Tests for Tag dataclass."""

    def test_tag_creation(self):
        """Test basic tag creation."""
        tag = Tag(label="dog", confidence=0.95)
        assert tag.label == "dog"
        assert tag.confidence == 0.95

    def test_tag_to_dict(self):
        """Test tag serialization."""
        tag = Tag(label="cat", confidence=0.87)
        result = tag.to_dict()
        assert result == {"label": "cat", "confidence": 0.87}

    def test_tag_confidence_bounds(self):
        """Test tag with edge confidence values."""
        tag_low = Tag(label="test", confidence=0.0)
        tag_high = Tag(label="test", confidence=1.0)
        assert tag_low.confidence == 0.0
        assert tag_high.confidence == 1.0


class TestTagResult:
    """Tests for TagResult dataclass."""

    def test_tag_result_creation(self):
        """Test basic TagResult creation."""
        tags = [Tag(label="dog", confidence=0.95)]
        result = TagResult(
            tags=tags,
            model="test_model",
            version="1.0.0",
            inference_time_ms=100.5,
        )
        assert result.model == "test_model"
        assert result.version == "1.0.0"
        assert result.inference_time_ms == 100.5
        assert len(result.tags) == 1

    def test_tag_result_empty_tags(self):
        """Test TagResult with no tags."""
        result = TagResult(
            tags=[],
            model="test_model",
            version="1.0.0",
            inference_time_ms=50.0,
        )
        assert result.tags == []

    def test_tag_result_to_dict(self):
        """Test TagResult serialization."""
        tags = [
            Tag(label="dog", confidence=0.95),
            Tag(label="outdoor", confidence=0.80),
        ]
        result = TagResult(
            tags=tags,
            model="ram_plus",
            version="1.0.0",
            inference_time_ms=142.5,
        )
        output = result.to_dict()

        assert output["model"] == "ram_plus"
        assert output["version"] == "1.0.0"
        assert output["inference_time_ms"] == 142.5
        assert len(output["tags"]) == 2
        assert output["tags"][0]["label"] == "dog"
        assert output["tags"][1]["confidence"] == 0.80


class TestPluginInfo:
    """Tests for PluginInfo dataclass."""

    def test_plugin_info_creation(self):
        """Test basic PluginInfo creation."""
        info = PluginInfo(
            name="test_plugin",
            version="1.0.0",
            description="A test plugin",
        )
        assert info.name == "test_plugin"
        assert info.version == "1.0.0"
        assert info.description == "A test plugin"
        assert info.hardware_reqs == {}

    def test_plugin_info_with_hardware_reqs(self):
        """Test PluginInfo with hardware requirements."""
        info = PluginInfo(
            name="gpu_plugin",
            version="2.0.0",
            description="Needs GPU",
            hardware_reqs={"gpu": True, "min_vram_gb": 4},
        )
        assert info.hardware_reqs["gpu"] is True
        assert info.hardware_reqs["min_vram_gb"] == 4

    def test_plugin_info_to_dict(self):
        """Test PluginInfo serialization."""
        info = PluginInfo(
            name="ram_plus",
            version="1.0.0",
            description="RAM++ plugin",
            hardware_reqs={"min_ram_gb": 4},
        )
        output = info.to_dict()

        assert output["name"] == "ram_plus"
        assert output["version"] == "1.0.0"
        assert output["description"] == "RAM++ plugin"
        assert output["hardware_reqs"]["min_ram_gb"] == 4


class TestHardwareProfile:
    """Tests for HardwareProfile dataclass."""

    def test_hardware_profile_cpu_only(self):
        """Test HardwareProfile for CPU-only system."""
        profile = HardwareProfile(
            cpu_model="Intel Core i7-10700",
            cpu_cores=8,
            ram_total_gb=16.0,
            ram_available_gb=12.0,
        )
        assert profile.cpu_model == "Intel Core i7-10700"
        assert profile.cpu_cores == 8
        assert profile.gpu_type is None
        assert profile.gpu_name is None
        assert profile.gpu_vram_gb is None

    def test_hardware_profile_with_gpu(self):
        """Test HardwareProfile with GPU."""
        profile = HardwareProfile(
            cpu_model="AMD Ryzen 9 5900X",
            cpu_cores=12,
            ram_total_gb=64.0,
            ram_available_gb=48.0,
            gpu_type="cuda",
            gpu_name="NVIDIA RTX 3080",
            gpu_vram_gb=10.0,
        )
        assert profile.gpu_type == "cuda"
        assert profile.gpu_name == "NVIDIA RTX 3080"
        assert profile.gpu_vram_gb == 10.0

    def test_hardware_profile_to_dict(self):
        """Test HardwareProfile serialization."""
        profile = HardwareProfile(
            cpu_model="Apple M2 Pro",
            cpu_cores=12,
            ram_total_gb=32.0,
            ram_available_gb=24.0,
            gpu_type="mps",
            gpu_name="Apple M2 Pro",
        )
        output = profile.to_dict()

        assert output["cpu_model"] == "Apple M2 Pro"
        assert output["cpu_cores"] == 12
        assert output["ram_total_gb"] == 32.0
        assert output["ram_available_gb"] == 24.0
        assert output["gpu_type"] == "mps"
        assert output["gpu_name"] == "Apple M2 Pro"
        assert output["gpu_vram_gb"] is None
