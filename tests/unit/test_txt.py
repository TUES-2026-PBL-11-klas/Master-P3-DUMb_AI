"""
Unit tests for TxtParser.

Run with:
    pytest test_txt.py -v
"""

from __future__ import annotations

import uuid

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
def txt_bytes() -> bytes:
    """A representative .txt payload with multi-line content."""
    return "Line one\nLine two\nLine three\n".encode("utf-8")


@pytest.fixture
def plain_bytes() -> bytes:
    """Minimal .txt payload with just plain text."""
    return "Hello world.\n".encode("utf-8")


# Basic parsing


class TestTxtParserBasic:
    def test_returns_document(self, parser: TxtParser, txt_bytes: bytes) -> None:
        result = parser.parse(txt_bytes, "notes.txt")

        assert isinstance(result, Document)
        assert isinstance(result.id, uuid.UUID)
        assert result.filename == "notes.txt"
        assert result.user_id == uuid.UUID(int=0)
        assert len(result.content) > 0

    def test_content_ends_with_newline(
        self, parser: TxtParser, txt_bytes: bytes
    ) -> None:
        doc = parser.parse(txt_bytes, "notes.txt")
        assert doc.content.endswith("\n")

    def test_plain_txt_round_trip(self, parser: TxtParser, plain_bytes: bytes) -> None:
        doc = parser.parse(plain_bytes, "plain.txt")
        assert "Hello world." in doc.content

    def test_content_matches_payload(self, parser: TxtParser, txt_bytes: bytes) -> None:
        doc = parser.parse(txt_bytes, "notes.txt")
        assert "Line one" in doc.content
        assert "Line two" in doc.content
        assert "Line three" in doc.content

    def test_id_is_uuid(self, parser: TxtParser, txt_bytes: bytes) -> None:
        doc = parser.parse(txt_bytes, "notes.txt")
        assert isinstance(doc.id, uuid.UUID)

    def test_user_id_is_nil_sentinel(self, parser: TxtParser, txt_bytes: bytes) -> None:
        # TxtParser sets a nil UUID; IngestionService replaces it later.
        doc = parser.parse(txt_bytes, "notes.txt")
        assert doc.user_id == uuid.UUID(int=0)

    def test_bytearray_accepted(self, parser: TxtParser) -> None:
        """parse() accepts both bytes and bytearray for caller flexibility."""
        doc = parser.parse(bytearray(b"hi there\n"), "hi.txt")
        assert "hi there" in doc.content


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
    def test_trailing_spaces_stripped(self, parser: TxtParser) -> None:
        doc = parser.parse(b"line one   \nline two\t\t\n", "notes.txt")
        for line in doc.content.splitlines():
            assert line == line.rstrip(), f"Trailing whitespace found: {line!r}"

    def test_excessive_blank_lines_collapsed(self, parser: TxtParser) -> None:
        doc = parser.parse(b"a\n\n\n\n\nb\n", "notes.txt")
        assert "\n\n\n" not in doc.content

    def test_empty_payload_produces_single_newline(self, parser: TxtParser) -> None:
        doc = parser.parse(b"", "empty.txt")
        assert doc.content == "\n"


# Encoding fallback


class TestTxtParserEncoding:
    def test_latin1_payload(self, parser: TxtParser) -> None:
        # 0xe9 ('é' in latin-1) is invalid as standalone UTF-8.
        doc = parser.parse(b"caf\xe9\n", "legacy.txt")
        assert "caf" in doc.content  # decoded without crashing

    def test_utf8_bom(self, parser: TxtParser) -> None:
        doc = parser.parse(b"\xef\xbb\xbfhello\n", "bom.txt")
        assert "hello" in doc.content

    def test_utf16(self, parser: TxtParser) -> None:
        doc = parser.parse("hello world\n".encode("utf-16"), "wide.txt")
        assert "hello world" in doc.content


# Error cases


class TestTxtParserErrors:
    def test_wrong_extension_raises(self, parser: TxtParser) -> None:
        with pytest.raises(UnsupportedFormatError, match="does not handle"):
            parser.parse(b"# heading\n", "notes.md")

    def test_wrong_extension_pdf(self, parser: TxtParser) -> None:
        with pytest.raises(UnsupportedFormatError):
            parser.parse(b"fake pdf\n", "notes.pdf")

    def test_non_bytes_raises(self, parser: TxtParser) -> None:
        """parse() guards against accidental str input."""
        with pytest.raises(RAGException, match="expected bytes"):
            parser.parse("not bytes", "notes.txt")  # type: ignore[arg-type]

    def test_filename_with_path_component(self, parser: TxtParser) -> None:
        """A client that includes a leading dir should still get its extension parsed."""
        doc = parser.parse(b"hi\n", "subdir/notes.txt")
        assert isinstance(doc, Document)


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
