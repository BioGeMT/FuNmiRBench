"""Tests for funmirbench.utils.paths."""

import pathlib

import pytest

from funmirbench.utils.paths import project_root, resolve_path, root_relative_path


class TestProjectRoot:
    def test_override_takes_precedence(self, tmp_path):
        result = project_root(tmp_path)
        assert result == tmp_path.resolve()

    def test_env_var_takes_precedence(self, tmp_path, monkeypatch):
        monkeypatch.setenv("FUNMIRBENCH_ROOT", str(tmp_path))
        result = project_root()
        assert result == tmp_path.resolve()

    def test_marker_search_finds_repo(self):
        # Running from within the real repo — should find the actual root.
        root = project_root()
        assert (root / "pyproject.toml").is_file()


class TestResolvePath:
    def test_relative_joined_with_root(self, tmp_path):
        result = resolve_path(tmp_path, pathlib.Path("foo/bar.tsv"))
        assert result == (tmp_path / "foo" / "bar.tsv").resolve()

    def test_absolute_returned_as_is(self, tmp_path):
        abs_path = tmp_path / "abs.tsv"
        result = resolve_path(tmp_path, abs_path)
        assert result == abs_path.resolve()


class TestRootRelativePath:
    def test_relative_unchanged(self, tmp_path):
        rel = pathlib.Path("data/foo.tsv")
        result = root_relative_path(tmp_path, rel)
        assert result == rel

    def test_absolute_under_root_converted(self, tmp_path):
        abs_path = tmp_path / "data" / "foo.tsv"
        abs_path.parent.mkdir(parents=True, exist_ok=True)
        abs_path.touch()
        result = root_relative_path(tmp_path, abs_path)
        assert result == pathlib.Path("data/foo.tsv")

    def test_absolute_outside_root_raises(self, tmp_path):
        outside = pathlib.Path("/some/other/place")
        with pytest.raises(ValueError, match="repo-relative"):
            root_relative_path(tmp_path, outside)
