"""
Unit tests for TxtParser.

Run with:
    pytest test_txt.py -v
"""

from __future__ import annotations

import uuid
from pathlib import Path

import pytest

from services.ingestion.parsers.txt import TxtParser
from services.ingestion.parsers.registry import ParserRegistry
from services.shared.domain import Document
from services.shared.exceptions import RAGException, UnsupportedFormatError


# Fixtures


@pytest.fixture
def parser() -> TxtParser:
    return TxtParser()


@pytest.fixture
def txt_file(tmp_path: Path) -> Path:
    """A representative .txt file with multi-line content."""
    p = tmp_path / "notes.txt"
    p.write_text("Line one\nLine two\nLine three\n", encoding="utf-8")
    return p


@pytest.fixture
def plain_txt(tmp_path: Path) -> Path:
    """Minimal .txt file with just plain text."""
    p = tmp_path / "plain.txt"
    p.write_text("Hello world.\n", encoding="utf-8")
    return p


# Basic parsing


class TestTxtParserBasic:
    def test_returns_document(self, parser: TxtParser, txt_file: Path) -> None:
        result = parser.parse(txt_file)

        assert isinstance(result, Document)
        assert isinstance(result.id, uuid.UUID)
        assert result.filename == "notes.txt"
        assert result.user_id == uuid.UUID(int=0)
        assert len(result.content) > 0

    def test_content_ends_with_newline(self, parser: TxtParser, txt_file: Path) -> None:
        doc = parser.parse(txt_file)
        assert doc.content.endswith("\n")

    def test_plain_txt_round_trip(self, parser: TxtParser, plain_txt: Path) -> None:
        doc = parser.parse(plain_txt)
        assert "Hello world." in doc.content

    def test_content_matches_file(self, parser: TxtParser, txt_file: Path) -> None:
        doc = parser.parse(txt_file)
        assert "Line one" in doc.content
        assert "Line two" in doc.content
        assert "Line three" in doc.content

    def test_id_is_uuid(self, parser: TxtParser, txt_file: Path) -> None:
        doc = parser.parse(txt_file)
        assert isinstance(doc.id, uuid.UUID)

    def test_user_id_is_nil_sentinel(self, parser: TxtParser, txt_file: Path) -> None:
        # TxtParser sets a nil UUID; IngestionService replaces it later.
        doc = parser.parse(txt_file)
        assert doc.user_id == uuid.UUID(int=0)


# Extension declarations


class TestTxtParserExtensions:
    def test_extensions_attribute(self, parser: TxtParser) -> None:
        assert "txt" in parser.extensions

    def test_does_not_declare_md(self, parser: TxtParser) -> None:
        assert "md" not in parser.extensions

    def test_does_not_declare_pdf(self, parser: TxtParser) -> None:
        assert "pdf" not in parser.extensions

    def test_extensions_is_tuple(self, parser: TxtParser) -> None:
        assert isinstance(parser.extensions, tuple)


# Whitespace normalisation


class TestTxtParserNormalisation:
    def test_trailing_spaces_stripped(self, parser: TxtParser, tmp_path: Path) -> None:
        f = tmp_path / "notes.txt"
        f.write_text("line one   \nline two\t\t\n", encoding="utf-8")
        doc = parser.parse(f)
        for line in doc.content.splitlines():
            assert line == line.rstrip(), f"Trailing whitespace found: {line!r}"

    def test_excessive_blank_lines_collapsed(
        self, parser: TxtParser, tmp_path: Path
    ) -> None:
        f = tmp_path / "notes.txt"
        f.write_text("a\n\n\n\n\nb\n", encoding="utf-8")
        doc = parser.parse(f)
        assert "\n\n\n" not in doc.content

    def test_empty_file_produces_single_newline(
        self, parser: TxtParser, tmp_path: Path
    ) -> None:
        f = tmp_path / "empty.txt"
        f.write_text("", encoding="utf-8")
        doc = parser.parse(f)
        assert doc.content == "\n"


# Encoding fallback


class TestTxtParserEncoding:
    def test_latin1_file(self, parser: TxtParser, tmp_path: Path) -> None:
        f = tmp_path / "legacy.txt"
        # 0xe9 ('é' in latin-1) is invalid as standalone UTF-8.
        f.write_bytes(b"caf\xe9\n")
        doc = parser.parse(f)
        assert "caf" in doc.content  # read without crashing

    def test_utf8_bom(self, parser: TxtParser, tmp_path: Path) -> None:
        f = tmp_path / "bom.txt"
        f.write_bytes(b"\xef\xbb\xbfhello\n")
        doc = parser.parse(f)
        assert "hello" in doc.content

    def test_utf16(self, parser: TxtParser, tmp_path: Path) -> None:
        f = tmp_path / "wide.txt"
        f.write_bytes("hello world\n".encode("utf-16"))
        doc = parser.parse(f)
        assert "hello world" in doc.content


# Error cases


class TestTxtParserErrors:
    def test_missing_file_raises(self, parser: TxtParser, tmp_path: Path) -> None:
        with pytest.raises(RAGException, match="does not exist"):
            parser.parse(tmp_path / "nonexistent.txt")

    def test_directory_raises(self, parser: TxtParser, tmp_path: Path) -> None:
        with pytest.raises(RAGException, match="not a regular file"):
            parser.parse(tmp_path)

    def test_wrong_extension_raises(self, parser: TxtParser, tmp_path: Path) -> None:
        f = tmp_path / "notes.md"
        f.write_text("# heading\n", encoding="utf-8")
        with pytest.raises(UnsupportedFormatError, match="does not handle"):
            parser.parse(f)

    def test_wrong_extension_pdf(self, parser: TxtParser, tmp_path: Path) -> None:
        f = tmp_path / "notes.pdf"
        f.write_text("fake pdf\n", encoding="utf-8")
        with pytest.raises(UnsupportedFormatError):
            parser.parse(f)


# ParserRegistry integration


class TestRegistryIntegration:
    def test_registry_finds_txt(self, parser: TxtParser) -> None:
        """TxtParser can be registered and looked up by extension."""
        reg: ParserRegistry[Document] = ParserRegistry()
        reg.register(parser)

        found = reg.get("txt")
        assert found is parser

    def test_registry_finds_txt_with_dot(self, parser: TxtParser) -> None:
        reg: ParserRegistry[Document] = ParserRegistry()
        reg.register(parser)

        found = reg.get(".txt")
        assert found is parser

    def test_registry_supported_extensions(self, parser: TxtParser) -> None:
        reg: ParserRegistry[Document] = ParserRegistry()
        reg.register(parser)

        exts = reg.supported_extensions
        assert "txt" in exts
