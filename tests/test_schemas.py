"""Tests for plugin schemas."""

import math

from visual_buffet.plugins.schemas import (
    HardwareProfile,
    MergedTag,
    PluginInfo,
    Tag,
    TagResult,
    boost_confidence,
    merge_tags,
)


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


class TestBoostConfidence:
    """Tests for boost_confidence function."""

    def test_no_boost_single_source(self):
        """Single source should not boost."""
        result = boost_confidence(0.7, sources=1)
        assert result == 0.7

    def test_no_boost_zero_confidence(self):
        """Zero confidence should not boost."""
        result = boost_confidence(0.0, sources=5)
        assert result == 0.0

    def test_no_boost_one_confidence(self):
        """Full confidence should not boost."""
        result = boost_confidence(1.0, sources=5)
        assert result == 1.0

    def test_boost_multiple_sources(self):
        """Multiple sources should boost confidence."""
        raw = 0.7
        boosted = boost_confidence(raw, sources=5)
        assert boosted > raw
        assert boosted < 1.0

    def test_boost_expected_values(self):
        """Test expected boost values from documentation."""
        # 0.70 with 5 sources -> ~0.81 (4 additional sources * 0.15 = 0.6 log-odds boost)
        result = boost_confidence(0.70, sources=5)
        assert 0.80 < result < 0.83

        # 0.85 with 5 sources -> ~0.91
        result = boost_confidence(0.85, sources=5)
        assert 0.90 < result < 0.93

        # 0.95 with 5 sources -> ~0.97
        result = boost_confidence(0.95, sources=5)
        assert 0.96 < result < 0.98

    def test_boost_diminishing_returns(self):
        """High confidence should boost less than low confidence."""
        low_raw = 0.5
        high_raw = 0.9

        low_boosted = boost_confidence(low_raw, sources=5)
        high_boosted = boost_confidence(high_raw, sources=5)

        low_delta = low_boosted - low_raw
        high_delta = high_boosted - high_raw

        # Low confidence should get bigger absolute boost
        assert low_delta > high_delta

    def test_boost_custom_boost_per_source(self):
        """Test custom boost_per_source parameter."""
        raw = 0.7
        default_boost = boost_confidence(raw, sources=5, boost_per_source=0.15)
        high_boost = boost_confidence(raw, sources=5, boost_per_source=0.25)
        low_boost = boost_confidence(raw, sources=5, boost_per_source=0.05)

        assert high_boost > default_boost > low_boost

    def test_boost_stays_bounded(self):
        """Boosted confidence should never exceed 1.0."""
        result = boost_confidence(0.99, sources=10, boost_per_source=0.5)
        assert result < 1.0


class TestMergedTag:
    """Tests for MergedTag dataclass."""

    def test_merged_tag_creation(self):
        """Test basic MergedTag creation."""
        tag = MergedTag(
            label="dog",
            raw_confidence=0.85,
            boosted_confidence=0.92,
            sources=5,
            max_sources=5,
        )
        assert tag.label == "dog"
        assert tag.raw_confidence == 0.85
        assert tag.boosted_confidence == 0.92
        assert tag.sources == 5
        assert tag.max_sources == 5

    def test_merged_tag_confidence_property(self):
        """Test confidence property returns boosted when available."""
        tag_boosted = MergedTag(
            label="cat",
            raw_confidence=0.70,
            boosted_confidence=0.82,
            sources=5,
            max_sources=5,
        )
        assert tag_boosted.confidence == 0.82

        tag_raw_only = MergedTag(
            label="dog",
            raw_confidence=0.90,
            boosted_confidence=None,
            sources=1,
            max_sources=5,
        )
        assert tag_raw_only.confidence == 0.90

    def test_merged_tag_to_dict_with_boost(self):
        """Test serialization includes both raw and boosted."""
        tag = MergedTag(
            label="sunset",
            raw_confidence=0.85,
            boosted_confidence=0.92,
            sources=5,
            max_sources=5,
        )
        result = tag.to_dict()

        assert result["label"] == "sunset"
        assert result["confidence"] == 0.92
        assert result["raw_confidence"] == 0.85
        assert result["sources"] == 5
        assert result["max_sources"] == 5

    def test_merged_tag_to_dict_single_source(self):
        """Test serialization for single source (no boost)."""
        tag = MergedTag(
            label="cat",
            raw_confidence=0.95,
            boosted_confidence=0.95,  # Same as raw for single source
            sources=1,
            max_sources=5,
        )
        result = tag.to_dict()

        assert result["label"] == "cat"
        assert result["confidence"] == 0.95
        assert "raw_confidence" in result  # Still included since boosted != None
        assert result["sources"] == 1
        assert result["max_sources"] == 5


class TestMergeTags:
    """Tests for merge_tags function."""

    def test_merge_duplicate_tags(self):
        """Test merging duplicate tags from multiple resolutions."""
        tags = [
            Tag("dog", 0.90),
            Tag("dog", 0.85),
            Tag("cat", 0.70),
        ]
        merged = merge_tags(tags, resolutions_used=[480, 1080])

        # Should have 2 unique tags
        assert len(merged) == 2

        # Dog should have sources=2, cat should have sources=1
        dog_tag = next(t for t in merged if t.label == "dog")
        cat_tag = next(t for t in merged if t.label == "cat")

        assert dog_tag.sources == 2
        assert cat_tag.sources == 1

    def test_merge_keeps_highest_raw_confidence(self):
        """Test that raw_confidence is the max across resolutions."""
        tags = [
            Tag("dog", 0.90),
            Tag("dog", 0.85),
            Tag("dog", 0.92),
        ]
        merged = merge_tags(tags, resolutions_used=[480, 1080, 2048])

        dog_tag = merged[0]
        assert dog_tag.raw_confidence == 0.92
        assert dog_tag.sources == 3

    def test_merge_calculates_boosted_confidence(self):
        """Test that boosted confidence is calculated."""
        tags = [
            Tag("dog", 0.85),
            Tag("dog", 0.85),
            Tag("dog", 0.85),
        ]
        merged = merge_tags(tags, resolutions_used=[480, 1080, 2048])

        dog_tag = merged[0]
        assert dog_tag.raw_confidence == 0.85
        assert dog_tag.boosted_confidence is not None
        assert dog_tag.boosted_confidence > dog_tag.raw_confidence

    def test_merge_single_source_no_boost(self):
        """Test that single source tags are not boosted."""
        tags = [Tag("cat", 0.95)]
        merged = merge_tags(tags, resolutions_used=[1080])

        cat_tag = merged[0]
        assert cat_tag.raw_confidence == 0.95
        assert cat_tag.boosted_confidence == 0.95  # No boost for single source

    def test_merge_sorts_by_boosted_confidence(self):
        """Test that results are sorted by boosted confidence."""
        tags = [
            # Cat has high raw but single source -> no boost
            Tag("cat", 0.95),
            # Dog has lower raw but 5 sources -> boosted higher
            Tag("dog", 0.70),
            Tag("dog", 0.70),
            Tag("dog", 0.70),
            Tag("dog", 0.70),
            Tag("dog", 0.70),
        ]
        merged = merge_tags(tags, resolutions_used=[480, 1080, 2048, 4096, 0])

        # Dog should rank higher due to boost (0.70 -> ~0.82)
        # Cat stays at 0.95 with no boost
        # So cat should still be first since 0.95 > 0.82
        assert merged[0].label == "cat"
        assert merged[1].label == "dog"

    def test_merge_max_sources_set_correctly(self):
        """Test that max_sources is set from resolutions_used."""
        tags = [Tag("dog", 0.80)]
        merged = merge_tags(tags, resolutions_used=[480, 1080, 2048, 4096, 0])

        assert merged[0].max_sources == 5

    def test_merge_case_insensitive(self):
        """Test that tag merging is case-insensitive."""
        tags = [
            Tag("Dog", 0.90),
            Tag("DOG", 0.85),
            Tag("dog", 0.80),
        ]
        merged = merge_tags(tags)

        assert len(merged) == 1
        assert merged[0].label == "dog"
        assert merged[0].sources == 3

    def test_merge_empty_labels_ignored(self):
        """Test that empty or whitespace labels are ignored."""
        tags = [
            Tag("dog", 0.90),
            Tag("", 0.50),
            Tag("  ", 0.50),
        ]
        merged = merge_tags(tags)

        assert len(merged) == 1
        assert merged[0].label == "dog"
