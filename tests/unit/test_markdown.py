"""
Unit tests for MarkdownParser.

Run with:
    pytest test_markdown.py -v
"""

from __future__ import annotations

import uuid
from pathlib import Path

import pytest

from services.ingestion.parsers.markdown import MarkdownParser
from services.ingestion.parsers.registry import ParserRegistry
from services.shared.domain import Document
from services.shared.exceptions import RAGException, UnsupportedFormatError


# Fixtures


@pytest.fixture
def parser() -> MarkdownParser:
    return MarkdownParser()


@pytest.fixture
def md_file(tmp_path: Path) -> Path:
    """Create a sample .md file with various Markdown elements."""
    content = """\
# Main Title

Some introductory paragraph with **bold text** and *italic text*.

## Section Two

Here is a [link to docs](https://example.com) and an image:

![alt text](https://example.com/img.png)

### Code Example

```python
def hello():
    print("world")
```

> This is a blockquote.
> It spans two lines.

- Item one
- Item two
  - Nested item

1. First
2. Second

---

Final paragraph with `inline code` and ~~strikethrough~~.
"""
    p = tmp_path / "notes.md"
    p.write_text(content, encoding="utf-8")
    return p


@pytest.fixture
def plain_md(tmp_path: Path) -> Path:
    """Minimal .md file with just plain text."""
    p = tmp_path / "plain.md"
    p.write_text("Hello world.\n", encoding="utf-8")
    return p


# Basic parsing


class TestMarkdownParserBasic:
    def test_returns_document(self, parser: MarkdownParser, md_file: Path) -> None:
        result = parser.parse(md_file)

        assert isinstance(result, Document)
        assert isinstance(result.id, uuid.UUID)
        assert result.filename == "notes.md"
        assert result.user_id == uuid.UUID(int=0)
        assert len(result.content) > 0

    def test_content_ends_with_newline(
        self, parser: MarkdownParser, md_file: Path
    ) -> None:
        doc = parser.parse(md_file)
        assert doc.content.endswith("\n")

    def test_plain_md_round_trip(self, parser: MarkdownParser, plain_md: Path) -> None:
        doc = parser.parse(plain_md)
        assert "Hello world." in doc.content

    def test_content_matches_file(self, parser: MarkdownParser, md_file: Path) -> None:
        doc = parser.parse(md_file)
        assert "Main Title" in doc.content
        assert "Section Two" in doc.content
        assert "Final paragraph" in doc.content

    def test_id_is_uuid(self, parser: MarkdownParser, md_file: Path) -> None:
        doc = parser.parse(md_file)
        assert isinstance(doc.id, uuid.UUID)

    def test_user_id_is_nil_sentinel(
        self, parser: MarkdownParser, md_file: Path
    ) -> None:
        # MarkdownParser sets a nil UUID; IngestionService replaces it later.
        doc = parser.parse(md_file)
        assert doc.user_id == uuid.UUID(int=0)


# Extension declaration


class TestMarkdownParserExtensions:
    def test_extensions_attribute(self, parser: MarkdownParser) -> None:
        assert "md" in parser.extensions
        assert "markdown" in parser.extensions

    def test_does_not_declare_txt(self, parser: MarkdownParser) -> None:
        assert "txt" not in parser.extensions

    def test_does_not_declare_pdf(self, parser: MarkdownParser) -> None:
        assert "pdf" not in parser.extensions

    def test_extensions_is_tuple(self, parser: MarkdownParser) -> None:
        assert isinstance(parser.extensions, tuple)

    def test_markdown_extension_accepted(
        self, parser: MarkdownParser, tmp_path: Path
    ) -> None:
        p = tmp_path / "notes.markdown"
        p.write_text("Test content.\n", encoding="utf-8")
        doc = parser.parse(p)
        assert doc.filename == "notes.markdown"


# Markdown stripping (no plain-text equivalent — markdown-only)


class TestMarkdownStripping:
    """Verify that Markdown syntax is removed and prose content preserved."""

    def test_headings_stripped(self, parser: MarkdownParser, md_file: Path) -> None:
        doc = parser.parse(md_file)
        assert "Main Title" in doc.content
        assert "Section Two" in doc.content
        assert "# " not in doc.content
        assert "## " not in doc.content

    def test_bold_unwrapped(self, parser: MarkdownParser, md_file: Path) -> None:
        doc = parser.parse(md_file)
        assert "bold text" in doc.content
        assert "**" not in doc.content

    def test_italic_unwrapped(self, parser: MarkdownParser, md_file: Path) -> None:
        doc = parser.parse(md_file)
        assert "italic text" in doc.content
        assert "*italic text*" not in doc.content

    def test_link_replaced_with_text(
        self, parser: MarkdownParser, md_file: Path
    ) -> None:
        doc = parser.parse(md_file)
        assert "link to docs" in doc.content
        assert "https://example.com)" not in doc.content
        assert "](http" not in doc.content

    def test_image_replaced_with_alt(
        self, parser: MarkdownParser, md_file: Path
    ) -> None:
        doc = parser.parse(md_file)
        assert "alt text" in doc.content
        assert "![" not in doc.content
        assert "img.png" not in doc.content

    def test_fenced_code_content_preserved(
        self, parser: MarkdownParser, md_file: Path
    ) -> None:
        doc = parser.parse(md_file)
        assert "def hello():" in doc.content
        assert '    print("world")' in doc.content
        assert "```" not in doc.content

    def test_blockquote_markers_stripped(
        self, parser: MarkdownParser, md_file: Path
    ) -> None:
        doc = parser.parse(md_file)
        assert "This is a blockquote." in doc.content
        for line in doc.content.splitlines():
            assert not line.startswith("> ")

    def test_unordered_list_bullets_stripped(
        self, parser: MarkdownParser, md_file: Path
    ) -> None:
        doc = parser.parse(md_file)
        assert "Item one" in doc.content
        assert "Item two" in doc.content
        assert "- Item one" not in doc.content

    def test_ordered_list_numbers_stripped(
        self, parser: MarkdownParser, md_file: Path
    ) -> None:
        doc = parser.parse(md_file)
        assert "First" in doc.content
        assert "1. First" not in doc.content

    def test_horizontal_rule_removed(
        self, parser: MarkdownParser, md_file: Path
    ) -> None:
        doc = parser.parse(md_file)
        assert "---" not in doc.content

    def test_inline_code_unwrapped(self, parser: MarkdownParser, md_file: Path) -> None:
        doc = parser.parse(md_file)
        assert "inline code" in doc.content
        assert "`inline code`" not in doc.content

    def test_strikethrough_unwrapped(
        self, parser: MarkdownParser, md_file: Path
    ) -> None:
        doc = parser.parse(md_file)
        assert "strikethrough" in doc.content
        assert "~~" not in doc.content

    def test_html_tags_removed(self, parser: MarkdownParser, tmp_path: Path) -> None:
        p = tmp_path / "html.md"
        p.write_text("Some <b>bold</b> and <em>italic</em> HTML.\n", encoding="utf-8")
        doc = parser.parse(p)
        assert "bold" in doc.content
        assert "italic" in doc.content
        assert "<b>" not in doc.content
        assert "</b>" not in doc.content
        assert "<em>" not in doc.content
        assert "</em>" not in doc.content

    def test_reference_link_stripped(
        self, parser: MarkdownParser, tmp_path: Path
    ) -> None:
        content = "See [the docs][ref1] for details.\n\n[ref1]: https://example.com\n"
        p = tmp_path / "ref.md"
        p.write_text(content, encoding="utf-8")
        doc = parser.parse(p)
        assert "the docs" in doc.content
        assert "[ref1]" not in doc.content
        assert "https://example.com" not in doc.content


# Whitespace normalisation


class TestMarkdownParserNormalisation:
    def test_trailing_spaces_stripped(
        self, parser: MarkdownParser, tmp_path: Path
    ) -> None:
        p = tmp_path / "spaces.md"
        p.write_text("line one   \nline two\t\t\n", encoding="utf-8")
        doc = parser.parse(p)
        for line in doc.content.splitlines():
            assert line == line.rstrip()

    def test_excessive_blank_lines_collapsed(
        self, parser: MarkdownParser, tmp_path: Path
    ) -> None:
        p = tmp_path / "blanks.md"
        p.write_text("para one\n\n\n\n\npara two\n", encoding="utf-8")
        doc = parser.parse(p)
        assert "\n\n\n" not in doc.content

    def test_empty_file_produces_single_newline(
        self, parser: MarkdownParser, tmp_path: Path
    ) -> None:
        p = tmp_path / "empty.md"
        p.write_text("", encoding="utf-8")
        doc = parser.parse(p)
        assert doc.content == "\n"


# Encoding fallback


class TestMarkdownParserEncoding:
    def test_latin1_file(self, parser: MarkdownParser, tmp_path: Path) -> None:
        p = tmp_path / "latin.md"
        p.write_bytes("Héllo café\n".encode("latin-1"))
        doc = parser.parse(p)
        assert "caf" in doc.content  # at least the ASCII portion survives

    def test_utf8_bom(self, parser: MarkdownParser, tmp_path: Path) -> None:
        p = tmp_path / "bom.md"
        p.write_bytes(b"\xef\xbb\xbf# Title\nContent.\n")
        doc = parser.parse(p)
        assert "Title" in doc.content

    def test_utf16(self, parser: MarkdownParser, tmp_path: Path) -> None:
        p = tmp_path / "wide.md"
        p.write_bytes("# Title\nhello world\n".encode("utf-16"))
        doc = parser.parse(p)
        assert "hello world" in doc.content


# Error cases


class TestMarkdownParserErrors:
    def test_missing_file_raises(self, parser: MarkdownParser, tmp_path: Path) -> None:
        with pytest.raises(RAGException, match="does not exist"):
            parser.parse(tmp_path / "nonexistent.md")

    def test_directory_raises(self, parser: MarkdownParser, tmp_path: Path) -> None:
        with pytest.raises(RAGException, match="not a regular file"):
            parser.parse(tmp_path)

    def test_wrong_extension_raises(
        self, parser: MarkdownParser, tmp_path: Path
    ) -> None:
        p = tmp_path / "notes.txt"
        p.write_text("hello\n", encoding="utf-8")
        with pytest.raises(UnsupportedFormatError, match="does not handle"):
            parser.parse(p)

    def test_wrong_extension_pdf(self, parser: MarkdownParser, tmp_path: Path) -> None:
        p = tmp_path / "notes.pdf"
        p.write_text("fake pdf\n", encoding="utf-8")
        with pytest.raises(UnsupportedFormatError):
            parser.parse(p)


# ParserRegistry integration


class TestRegistryIntegration:
    def test_registry_finds_md(self, parser: MarkdownParser) -> None:
        """MarkdownParser can be registered and looked up by extension."""
        reg: ParserRegistry[Document] = ParserRegistry()
        reg.register(parser)

        found = reg.get("md")
        assert found is parser

    def test_registry_finds_markdown(self, parser: MarkdownParser) -> None:
        reg: ParserRegistry[Document] = ParserRegistry()
        reg.register(parser)

        found = reg.get("markdown")
        assert found is parser

    def test_registry_supported_extensions(self, parser: MarkdownParser) -> None:
        reg: ParserRegistry[Document] = ParserRegistry()
        reg.register(parser)

        exts = reg.supported_extensions
        assert "md" in exts
        assert "markdown" in exts
