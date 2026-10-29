"""Unit tests for the sidecar URI helpers and document-path canonicalization
introduced when ``full_docs`` collapsed its four path fields to
``file_path`` + ``sidecar_location``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from urllib.parse import urlsplit

import pytest

from lightrag.utils_pipeline import (
    SIDECAR_LOCATION_UNKNOWN,
    normalize_document_file_path,
    parsed_artifact_dir_for,
    resolve_sidecar_uri,
    sidecar_assets_dir_for_uri,
    sidecar_blocks_path,
    sidecar_modality_path,
    sidecar_uri_for,
)


@pytest.mark.offline
def test_normalize_strips_hint_and_directory():
    assert normalize_document_file_path("abc.[native-iet].docx") == "abc.docx"
    assert normalize_document_file_path("/tmp/sub/abc.docx") == "abc.docx"
    assert normalize_document_file_path("abc.docx") == "abc.docx"


@pytest.mark.offline
def test_normalize_idempotent():
    once = normalize_document_file_path("/tmp/abc.[native].docx")
    twice = normalize_document_file_path(once)
    assert once == twice == "abc.docx"


@pytest.mark.offline
@pytest.mark.parametrize(
    "value",
    ["", None, "no-file-path", "unknown_source", "  "],
)
def test_normalize_maps_placeholders_to_unknown(value):
    assert normalize_document_file_path(value) == "unknown_source"


@pytest.mark.offline
def test_sidecar_uri_round_trip_ascii(tmp_path):
    sidecar_dir = tmp_path / "abc.docx.parsed"
    sidecar_dir.mkdir()
    uri = sidecar_uri_for(sidecar_dir)
    assert uri.startswith("file://")
    assert uri.endswith("/")
    resolved = resolve_sidecar_uri(uri)
    assert resolved == sidecar_dir.resolve()


@pytest.mark.offline
def test_sidecar_uri_round_trip_unicode_and_spaces(tmp_path):
    sidecar_dir = tmp_path / "中文 报告.docx.parsed"
    sidecar_dir.mkdir()
    uri = sidecar_uri_for(sidecar_dir)
    assert uri.startswith("file://")
    assert " " not in uri  # spaces are percent-encoded
    resolved = resolve_sidecar_uri(uri)
    assert resolved == sidecar_dir.resolve()


@pytest.mark.offline
def test_sidecar_uri_has_empty_authority(tmp_path):
    sidecar_dir = tmp_path / "demo.parsed"
    sidecar_dir.mkdir()
    uri = sidecar_uri_for(sidecar_dir)
    from urllib.parse import urlsplit

    assert urlsplit(uri).netloc == ""
    assert urlsplit(uri).path.startswith("/")


@pytest.mark.offline
def test_resolve_sidecar_uri_recovers_windows_drive_letter():
    uri = "file:///C:/Users/qhx/data/__parsed__/demo.pdf.parsed/"
    resolved = resolve_sidecar_uri(uri)
    assert resolved is not None
    # Normalizing separators makes the assertion hold on both POSIX
    # (where the drive path stays a relative string) and Windows.
    assert str(resolved).replace("\\", "/") == (
        "C:/Users/qhx/data/__parsed__/demo.pdf.parsed"
    )


@pytest.mark.offline
def test_resolve_sidecar_uri_recovers_legacy_writer_uri():
    # The previous writer stored the whole Windows drive path in the
    # authority, so the path component was just "/".
    uri = "file://C%3A%5CUsers%5Cqhx%5Cdata%5C__parsed__%5Cdemo.pdf.parsed/"
    assert str(resolve_sidecar_uri(uri)) == (
        r"C:\Users\qhx\data\__parsed__\demo.pdf.parsed"
    )


@pytest.mark.offline
def test_resolve_sidecar_uri_recovers_legacy_writer_uri_with_literal_escape():
    # Both halves of the defect at once: the previous writer put the whole drive
    # path in the authority, and the directory name itself looks percent-encoded.
    # The authority is decoded exactly once, so "%2520" must come back as the
    # literal text "%20" -- a second decode would silently point at "demo dir.parsed".
    uri = "file://C%3A%5Cdata%5Cdemo%2520dir.parsed/"
    resolved = resolve_sidecar_uri(uri)
    assert resolved is not None
    assert str(resolved).replace("\\", "/") == "C:/data/demo%20dir.parsed"


@pytest.mark.offline
def test_resolve_sidecar_uri_preserves_literal_escape_like_names():
    uri = "file:///C:/data/report%2520.parsed/"
    resolved = resolve_sidecar_uri(uri)
    assert resolved is not None
    # Decoded exactly once: "%2520" becomes the literal text "%20",
    # never a space.
    assert "%20" in str(resolved)
    assert "report .parsed" not in str(resolved)


@pytest.mark.offline
def test_resolve_sidecar_uri_tolerates_missing_trailing_slash(tmp_path):
    sidecar_dir = tmp_path / "demo.parsed"
    sidecar_dir.mkdir()
    uri_no_slash = sidecar_uri_for(sidecar_dir).rstrip("/")
    assert resolve_sidecar_uri(uri_no_slash) == sidecar_dir.resolve()


@pytest.mark.offline
@pytest.mark.parametrize(
    "uri",
    [None, "", SIDECAR_LOCATION_UNKNOWN, "s3://bucket/path/"],
)
def test_resolve_sidecar_uri_returns_none_for_unsupported(uri):
    assert resolve_sidecar_uri(uri) is None


@pytest.mark.offline
@pytest.mark.parametrize(
    "name",
    [
        "with space.docx.parsed",
        "中文 报告.docx.parsed",
        "literal%20lookalike.parsed",
        "literal%2Flookalike.parsed",
        "literal%25lookalike.parsed",
    ],
)
def test_sidecar_uri_round_trips_awkward_names(tmp_path, name):
    """A name that *looks* percent-encoded must survive the round trip.

    The reader has to decode exactly once: ``url2pathname`` already decodes, so
    unquoting first would rewrite the literal ``%20`` in ``literal%20lookalike``
    as a space and silently point at a different artifact directory.
    """
    sidecar_dir = tmp_path / name
    sidecar_dir.mkdir()
    uri = sidecar_uri_for(sidecar_dir)

    assert uri.endswith("/")
    # The bug this pins: a path must never end up in the URI authority, which
    # is what a hand-built ``file://`` + ``quote`` did with a drive letter.
    assert urlsplit(uri).netloc == ""
    assert resolve_sidecar_uri(uri) == sidecar_dir.resolve()


@pytest.mark.offline
@pytest.mark.parametrize(
    "name",
    [
        "C:/data/demo.parsed",
        "C:/data/a b.parsed",
        "C:/data/中文.parsed",
        "C:/data/report%20.parsed",
        "C:/data/a%2Fb.parsed",
        "C:/data/a%25b.parsed",
    ],
)
def test_windows_drive_paths_round_trip(name):
    """A drive-letter path must produce ``file:///C:/...``, not ``file://C%3A...``.

    Asserted from the URI shape rather than from ``as_uri`` so the contract is
    stated once: the authority is empty, the path is the path, and reading it
    back yields the same directory.
    """
    if sys.platform != "win32":
        pytest.skip("drive-letter paths are a Windows-only shape")
    sidecar_dir = Path(name)
    uri = sidecar_uri_for(sidecar_dir)

    assert uri.startswith("file:///")
    assert urlsplit(uri).netloc == ""
    assert resolve_sidecar_uri(uri) == sidecar_dir


@pytest.mark.offline
def test_sidecar_blocks_path_locates_jsonl(tmp_path):
    sidecar_dir = tmp_path / "demo.docx.parsed"
    sidecar_dir.mkdir()
    blocks = sidecar_dir / "demo.blocks.jsonl"
    blocks.write_text("", encoding="utf-8")
    uri = sidecar_uri_for(sidecar_dir)

    assert sidecar_blocks_path(uri) == str(blocks)
    assert sidecar_modality_path(uri, "tables") == str(sidecar_dir / "demo.tables.json")
    assert sidecar_assets_dir_for_uri(uri) == Path(sidecar_dir / "demo.blocks.assets")


@pytest.mark.offline
def test_sidecar_blocks_path_returns_none_when_missing(tmp_path):
    empty = tmp_path / "empty.parsed"
    empty.mkdir()
    uri = sidecar_uri_for(empty)
    assert sidecar_blocks_path(uri) is None
    assert sidecar_modality_path(uri, "drawings") is None
    assert sidecar_assets_dir_for_uri(uri) is None


@pytest.mark.offline
def test_parsed_artifact_dir_for_uses_input_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("INPUT_DIR", str(tmp_path))
    result = parsed_artifact_dir_for("demo.docx")
    assert result == tmp_path / "__parsed__" / "demo.docx.parsed"


@pytest.mark.offline
def test_parsed_artifact_dir_for_strips_hint(tmp_path, monkeypatch):
    monkeypatch.setenv("INPUT_DIR", str(tmp_path))
    result = parsed_artifact_dir_for("abc.[native-iet].docx")
    assert result == tmp_path / "__parsed__" / "abc.docx.parsed"
