import hashlib
import pathlib

import pytest

from funmirbench import experiment_store


class FakeResponse:
    def __init__(self, *, payload=None, chunks=None):
        self._payload = payload
        self._chunks = chunks or []

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload

    def iter_content(self, chunk_size):
        del chunk_size
        yield from self._chunks


def test_fetch_zenodo_file_registry_reads_record_listing(monkeypatch):
    seen = []

    def fake_get(url, timeout):
        seen.append((url, timeout))
        return FakeResponse(
            payload={
                "files": [
                    {
                        "key": "demo.tsv",
                        "size": 12,
                        "checksum": "md5:abc123",
                        "links": {"self": "https://zenodo.example/demo.tsv/content"},
                    }
                ]
            }
        )

    monkeypatch.setattr(experiment_store.requests, "get", fake_get)

    registry = experiment_store.fetch_zenodo_file_registry()

    assert registry == {
        "demo.tsv": {
            "filename": "demo.tsv",
            "size": 12,
            "checksum": "md5:abc123",
            "url": "https://zenodo.example/demo.tsv/content",
        }
    }
    assert seen == [(experiment_store.ZENODO_API_RECORD_URL, 120)]


def test_fetch_zenodo_file_registry_can_include_token(monkeypatch):
    seen = []

    def fake_get(url, timeout):
        seen.append((url, timeout))
        return FakeResponse(payload={"files": []})

    monkeypatch.setattr(experiment_store.requests, "get", fake_get)

    experiment_store.fetch_zenodo_file_registry(token="secret-token", timeout=30)

    assert seen == [
        (f"{experiment_store.ZENODO_API_RECORD_URL}?token=secret-token", 30)
    ]


def test_ensure_zenodo_experiment_cached_downloads_and_verifies_md5(tmp_path, monkeypatch):
    content = b"gene_id\tlogFC\tFDR\nENSG1\t1.0\t0.01\n"
    expected_md5 = hashlib.md5(content).hexdigest()
    seen = []

    def fake_get(url, stream, timeout):
        seen.append((url, stream, timeout))
        return FakeResponse(chunks=[content[:10], content[10:]])

    monkeypatch.setattr(experiment_store.requests, "get", fake_get)

    out = experiment_store.ensure_zenodo_experiment_cached(
        "data/experiments/processed/18745741/demo.tsv",
        repo=tmp_path,
        registry={
            "demo.tsv": {
                "filename": "demo.tsv",
                "checksum": f"md5:{expected_md5}",
                "url": "https://zenodo.example/demo.tsv/content",
            }
        },
    )

    assert out == (
        tmp_path / "data" / "experiments" / "processed" / "18745741" / "demo.tsv"
    )
    assert out.read_bytes() == content
    assert seen == [("https://zenodo.example/demo.tsv/content", True, 120)]
    assert list(out.parent.glob(".*demo.tsv.*.tmp")) == []


def test_ensure_zenodo_experiment_cached_reuses_valid_existing_file(tmp_path, monkeypatch):
    dest = tmp_path / "data" / "experiments" / "processed" / "18745741" / "demo.tsv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("ok\n", encoding="utf-8")
    expected_md5 = hashlib.md5(dest.read_bytes()).hexdigest()

    def fail_get(*args, **kwargs):
        raise AssertionError("network should not be used")

    monkeypatch.setattr(experiment_store.requests, "get", fail_get)

    out = experiment_store.ensure_zenodo_experiment_cached(
        "data/experiments/processed/18745741/demo.tsv",
        repo=tmp_path,
        registry={
            "demo.tsv": {
                "filename": "demo.tsv",
                "checksum": f"md5:{expected_md5}",
                "url": "https://zenodo.example/demo.tsv/content",
            }
        },
    )

    assert out == dest


def test_ensure_zenodo_experiment_cached_rejects_bad_existing_checksum(tmp_path, monkeypatch):
    dest = tmp_path / "data" / "experiments" / "processed" / "18745741" / "demo.tsv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("wrong\n", encoding="utf-8")

    def fail_get(*args, **kwargs):
        raise AssertionError("network should not be used")

    monkeypatch.setattr(experiment_store.requests, "get", fail_get)

    with pytest.raises(ValueError, match="failed checksum verification"):
        experiment_store.ensure_zenodo_experiment_cached(
            "data/experiments/processed/18745741/demo.tsv",
            repo=tmp_path,
            registry={
                "demo.tsv": {
                    "filename": "demo.tsv",
                    "checksum": "md5:00000000000000000000000000000000",
                    "url": "https://zenodo.example/demo.tsv/content",
                }
            },
        )


def test_ensure_zenodo_experiment_cached_force_keeps_existing_file_on_checksum_mismatch(tmp_path, monkeypatch):
    dest = tmp_path / "data" / "experiments" / "processed" / "18745741" / "demo.tsv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("good\n", encoding="utf-8")
    original = dest.read_text(encoding="utf-8")

    bad_content = b"bad-download\n"

    def fake_get(url, stream, timeout):
        assert url == "https://zenodo.example/demo.tsv/content"
        assert stream is True
        assert timeout == 120
        return FakeResponse(chunks=[bad_content])

    monkeypatch.setattr(experiment_store.requests, "get", fake_get)

    with pytest.raises(ValueError, match="Checksum mismatch"):
        experiment_store.ensure_zenodo_experiment_cached(
            "data/experiments/processed/18745741/demo.tsv",
            repo=tmp_path,
            force=True,
            registry={
                "demo.tsv": {
                    "filename": "demo.tsv",
                    "checksum": "md5:00000000000000000000000000000000",
                    "url": "https://zenodo.example/demo.tsv/content",
                }
            },
        )

    assert dest.read_text(encoding="utf-8") == original
    assert list(dest.parent.glob(".*demo.tsv.*.tmp")) == []


def test_ensure_zenodo_experiment_cached_force_keeps_existing_file_on_download_error(tmp_path, monkeypatch):
    dest = tmp_path / "data" / "experiments" / "processed" / "18745741" / "demo.tsv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("good\n", encoding="utf-8")
    original = dest.read_text(encoding="utf-8")

    class BrokenResponse(FakeResponse):
        def iter_content(self, chunk_size):
            del chunk_size
            yield b"partial"
            raise RuntimeError("download interrupted")

    def fake_get(url, stream, timeout):
        assert url == "https://zenodo.example/demo.tsv/content"
        assert stream is True
        assert timeout == 120
        return BrokenResponse()

    monkeypatch.setattr(experiment_store.requests, "get", fake_get)

    with pytest.raises(RuntimeError, match="download interrupted"):
        experiment_store.ensure_zenodo_experiment_cached(
            "data/experiments/processed/18745741/demo.tsv",
            repo=tmp_path,
            force=True,
            registry={
                "demo.tsv": {
                    "filename": "demo.tsv",
                    "checksum": "",
                    "url": "https://zenodo.example/demo.tsv/content",
                }
            },
        )

    assert dest.read_text(encoding="utf-8") == original
    assert list(dest.parent.glob(".*demo.tsv.*.tmp")) == []


def test_ensure_zenodo_experiment_cached_raises_for_unknown_filename(tmp_path):
    with pytest.raises(KeyError, match="not present in Zenodo record"):
        experiment_store.ensure_zenodo_experiment_cached(
            "data/experiments/processed/18745741/missing.tsv",
            repo=tmp_path,
            registry={},
        )


def test_sync_all_zenodo_experiments_uses_metadata_paths(tmp_path, monkeypatch):
    seen = []
    metadata_dir = tmp_path / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    (metadata_dir / "mirna_experiment_info.tsv").write_text(
        "id\tde_table_path\n"
        "dataset_b\tdata/experiments/processed/18745741/b.tsv\n"
        "dataset_a\tdata/experiments/processed/18745741/a.tsv\n",
        encoding="utf-8",
    )

    def fake_sync(paths, *, repo=None, registry=None, token=None, timeout=120, force=False):
        seen.append((paths, repo, registry, token, timeout, force))
        return [repo / pathlib.Path(path) for path in sorted(set(paths))]

    monkeypatch.setattr(experiment_store, "sync_zenodo_experiments", fake_sync)

    saved = experiment_store.sync_all_zenodo_experiments(
        repo=tmp_path,
        force=True,
    )

    assert saved == [
        tmp_path / "data" / "experiments" / "processed" / "18745741" / "a.tsv",
        tmp_path / "data" / "experiments" / "processed" / "18745741" / "b.tsv",
    ]
    assert seen == [
        (
            [
                "data/experiments/processed/18745741/b.tsv",
                "data/experiments/processed/18745741/a.tsv",
            ],
            tmp_path,
            None,
            None,
            120,
            True,
        ),
    ]


def test_sync_all_zenodo_experiments_deduplicates_repeated_metadata_paths(tmp_path, monkeypatch, capsys):
    seen = []
    metadata_dir = tmp_path / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    (metadata_dir / "mirna_experiment_info.tsv").write_text(
        "id\tde_table_path\n"
        "dataset_b1\tdata/experiments/processed/18745741/b.tsv\n"
        "dataset_a\tdata/experiments/processed/18745741/a.tsv\n"
        "dataset_b2\tdata/experiments/processed/18745741/b.tsv\n",
        encoding="utf-8",
    )

    def fake_ensure(path, *, repo=None, registry=None, token=None, timeout=120, force=False):
        seen.append((str(path), repo, registry, token, timeout, force))
        return repo / path

    registry = {
        "a.tsv": {"filename": "a.tsv", "checksum": "md5:a", "url": "https://zenodo/a"},
        "b.tsv": {"filename": "b.tsv", "checksum": "md5:b", "url": "https://zenodo/b"},
    }

    monkeypatch.setattr(
        experiment_store,
        "fetch_zenodo_file_registry",
        lambda *, token=None, timeout=120: registry,
    )
    monkeypatch.setattr(experiment_store, "ensure_zenodo_experiment_cached", fake_ensure)

    saved = experiment_store.sync_all_zenodo_experiments(repo=tmp_path)

    assert saved == [
        tmp_path / "data" / "experiments" / "processed" / "18745741" / "b.tsv",
        tmp_path / "data" / "experiments" / "processed" / "18745741" / "a.tsv",
    ]
    assert seen == [
        (
            str(pathlib.Path("data/experiments/processed/18745741/b.tsv")),
            tmp_path,
            registry,
            None,
            120,
            False,
        ),
        (
            str(pathlib.Path("data/experiments/processed/18745741/a.tsv")),
            tmp_path,
            registry,
            None,
            120,
            False,
        ),
    ]
    assert capsys.readouterr().out.splitlines() == ["sync b.tsv", "sync a.tsv"]


def test_sync_zenodo_experiments_syncs_only_selected_unique_paths(tmp_path, monkeypatch, capsys):
    seen = []
    registry = {
        "a.tsv": {"filename": "a.tsv", "checksum": "md5:a", "url": "https://zenodo/a"},
        "b.tsv": {"filename": "b.tsv", "checksum": "md5:b", "url": "https://zenodo/b"},
    }

    def fake_ensure(path, *, repo=None, registry=None, token=None, timeout=120, force=False):
        seen.append((str(path), repo, registry, token, timeout, force))
        return repo / path

    monkeypatch.setattr(experiment_store, "ensure_zenodo_experiment_cached", fake_ensure)

    saved = experiment_store.sync_zenodo_experiments(
        [
            "data/experiments/processed/18745741/b.tsv",
            "data/experiments/processed/18745741/a.tsv",
            "data/experiments/processed/18745741/b.tsv",
        ],
        repo=tmp_path,
        registry=registry,
    )

    assert saved == [
        tmp_path / "data" / "experiments" / "processed" / "18745741" / "b.tsv",
        tmp_path / "data" / "experiments" / "processed" / "18745741" / "a.tsv",
    ]
    assert seen == [
        (
            str(pathlib.Path("data/experiments/processed/18745741/b.tsv")),
            tmp_path,
            registry,
            None,
            120,
            False,
        ),
        (
            str(pathlib.Path("data/experiments/processed/18745741/a.tsv")),
            tmp_path,
            registry,
            None,
            120,
            False,
        ),
    ]
    assert capsys.readouterr().out.splitlines() == ["sync b.tsv", "sync a.tsv"]


def test_main_syncs_all_and_prints_summary(monkeypatch, capsys):
    monkeypatch.setattr(
        experiment_store,
        "parse_args",
        lambda: experiment_store.argparse.Namespace(
            repo=pathlib.Path("repo"),
            token="tok",
            timeout=90,
            force=True,
        ),
    )
    monkeypatch.setattr(
        experiment_store,
        "sync_all_zenodo_experiments",
        lambda **kwargs: [pathlib.Path("one.tsv"), pathlib.Path("two.tsv")],
    )

    experiment_store.main()

    assert capsys.readouterr().out.splitlines() == [
        "Synced 2 experiment tables into data/experiments/processed/18745741/"
    ]
