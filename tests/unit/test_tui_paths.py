"""
Unit tests for the TUI's path-input normalisation helper.

These cover the common Windows + cross-platform footguns that turn a
valid path into "file not found":
  - Surrounding double quotes (File Explorer's "Copy as path")
  - Surrounding single quotes (PowerShell tab completion)
  - Leading / trailing whitespace
  - A '~' for the home directory
"""

from __future__ import annotations

import os

from client.tui import normalize_input_path


# Quoted paths


def test_strips_surrounding_double_quotes() -> None:
    """File Explorer's 'Copy as path' wraps the result in double quotes."""
    assert normalize_input_path('"C:\\test.txt"') == os.path.normpath("C:\\test.txt")


def test_strips_surrounding_single_quotes() -> None:
    """PowerShell tab completion uses single quotes when the path has spaces."""
    assert normalize_input_path("'C:\\test.txt'") == os.path.normpath("C:\\test.txt")


def test_does_not_strip_mismatched_quotes() -> None:
    """A path that starts with a quote but doesn't end with one isn't quoted."""
    # The path is the literal string with the leading quote retained.
    assert normalize_input_path('"oddly_named.txt') == os.path.normpath(
        '"oddly_named.txt'
    )


def test_strips_quotes_plus_outer_whitespace() -> None:
    assert normalize_input_path('  "C:\\test.txt"  ') == os.path.normpath(
        "C:\\test.txt"
    )


def test_strips_whitespace_inside_quotes() -> None:
    """If the user accidentally typed spaces inside the quotes, clean them too."""
    assert normalize_input_path('"  C:\\test.txt  "') == os.path.normpath(
        "C:\\test.txt"
    )


# Whitespace


def test_strips_leading_and_trailing_whitespace() -> None:
    assert normalize_input_path("  /path/file.txt  ") == os.path.normpath(
        "/path/file.txt"
    )


def test_empty_string_stays_empty() -> None:
    assert normalize_input_path("") == ""


def test_whitespace_only_becomes_empty() -> None:
    assert normalize_input_path("   ") == ""


def test_empty_quotes_become_empty() -> None:
    assert normalize_input_path('""') == ""


def test_none_input_becomes_empty() -> None:
    assert normalize_input_path(None) == ""


# Home expansion


def test_tilde_expansion() -> None:
    result = normalize_input_path("~/notes.txt")
    # Whatever the home is on this platform, the result should not start
    # with a literal '~' anymore.
    assert not result.startswith("~")
    assert result.endswith("notes.txt") or result.endswith(
        "notes.txt".replace("/", os.sep)
    )


def test_tilde_alone() -> None:
    result = normalize_input_path("~")
    assert not result.startswith("~")


# Path normalisation


def test_collapses_double_slashes() -> None:
    assert normalize_input_path("/path//to///file.txt") == os.path.normpath(
        "/path/to/file.txt"
    )


def test_resolves_dot_dot() -> None:
    assert normalize_input_path("/a/b/../file.txt") == os.path.normpath("/a/file.txt")


def test_preserves_relative_path() -> None:
    assert normalize_input_path("notes.txt") == os.path.normpath("notes.txt")


# Plain paths shouldn't be mangled


def test_plain_absolute_unix_path_unchanged_apart_from_normpath() -> None:
    assert normalize_input_path("/home/user/file.txt") == os.path.normpath(
        "/home/user/file.txt"
    )


def test_plain_relative_subdir_path() -> None:
    assert normalize_input_path("subdir/notes.txt") == os.path.normpath(
        "subdir/notes.txt"
    )
