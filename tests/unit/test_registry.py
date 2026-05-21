"""
Unit tests for ParserRegistry.

Run with:
    pytest test_registry.py -v
"""

from __future__ import annotations

from pathlib import Path

import pytest

from services.ingestion.parsers.txt import TxtParser
from services.ingestion.parsers.registry import ParserRegistry
from services.shared.exceptions import UnsupportedFormatError


# Stub parsers


class _StubParser:
    """Minimal parser satisfying DocumentParser structurally."""

    extensions = ("xyz",)

    def parse(self, path: Path) -> str:
        return path.read_text()


class _AlsoTxt:
    """A second parser that also claims the 'txt' extension."""

    extensions = ("txt",)

    def parse(self, path: Path) -> str:
        return ""


# Fixtures


@pytest.fixture
def registry() -> ParserRegistry:
    return ParserRegistry()


# get() tests


class TestRegistryGet:
    def test_returns_correct_parser(self, registry: ParserRegistry) -> None:
        registry.register(TxtParser())
        assert isinstance(registry.get("txt"), TxtParser)

    def test_get_with_dot_prefix(self, registry: ParserRegistry) -> None:
        registry.register(TxtParser())
        assert isinstance(registry.get(".txt"), TxtParser)

    def test_get_is_case_insensitive(self, registry: ParserRegistry) -> None:
        registry.register(TxtParser())
        assert isinstance(registry.get("TXT"), TxtParser)

    def test_raises_for_unknown_extension(self, registry: ParserRegistry) -> None:
        with pytest.raises(UnsupportedFormatError):
            registry.get("pdf")

    def test_first_registered_wins(self, registry: ParserRegistry) -> None:
        """When two parsers claim the same extension, the first registered wins."""
        registry.register(TxtParser())
        registry.register(_AlsoTxt())
        assert isinstance(registry.get("txt"), TxtParser)


# get_for_path() tests


class TestRegistryGetForPath:
    def test_get_for_path(self, registry: ParserRegistry, tmp_path: Path) -> None:
        registry.register(TxtParser())
        f = tmp_path / "notes.txt"
        f.write_text("hi", encoding="utf-8")
        assert isinstance(registry.get_for_path(f), TxtParser)

    def test_no_extension_raises(
        self, registry: ParserRegistry, tmp_path: Path
    ) -> None:
        with pytest.raises(UnsupportedFormatError):
            registry.get_for_path(tmp_path / "README")

    def test_unregistered_extension_raises(
        self, registry: ParserRegistry, tmp_path: Path
    ) -> None:
        with pytest.raises(UnsupportedFormatError):
            registry.get_for_path(tmp_path / "data.pdf")


# Introspection


class TestRegistryIntrospection:
    def test_len_empty(self, registry: ParserRegistry) -> None:
        assert len(registry) == 0

    def test_len_after_register(self, registry: ParserRegistry) -> None:
        registry.register(TxtParser())
        assert len(registry) == 1
        registry.register(_StubParser())
        assert len(registry) == 2

    def test_repr_contains_parser_name(self, registry: ParserRegistry) -> None:
        registry.register(TxtParser())
        assert "TxtParser" in repr(registry)

    def test_supported_extensions_aggregated(self, registry: ParserRegistry) -> None:
        registry.register(TxtParser())
        registry.register(_StubParser())
        exts = registry.supported_extensions
        assert "txt" in exts
        assert "xyz" in exts

    def test_supported_extensions_sorted_and_deduped(
        self, registry: ParserRegistry
    ) -> None:
        registry.register(TxtParser())
        registry.register(_AlsoTxt())  # duplicate 'txt'
        exts = registry.supported_extensions
        assert exts == sorted(set(exts))


# Multiple parsers


class TestRegistryMultipleParsers:
    def test_dispatches_to_correct_parser(self, registry: ParserRegistry) -> None:
        registry.register(TxtParser())
        registry.register(_StubParser())
        assert isinstance(registry.get("txt"), TxtParser)
        assert isinstance(registry.get("xyz"), _StubParser)
