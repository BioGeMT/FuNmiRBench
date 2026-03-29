"""Tests for example GEO input downloads."""

import pathlib
from urllib.parse import parse_qs, urlparse

from funmirbench import geo_download_examples


class FakeResponse:
    def __init__(self, chunks):
        self._chunks = chunks

    def raise_for_status(self):
        return None

    def iter_content(self, chunk_size):
        del chunk_size
        yield from self._chunks


def test_download_examples_writes_selected_targets(tmp_path, monkeypatch):
    seen_urls = []

    def fake_get(url, stream, timeout):
        assert stream is True
        assert timeout == 120
        seen_urls.append(url)
        parsed = urlparse(url)
        query = parse_qs(parsed.query)
        name = query.get("file", [pathlib.Path(parsed.path).name])[0]
        return FakeResponse([f"content:{name}".encode("utf-8")])

    monkeypatch.setattr(geo_download_examples.requests, "get", fake_get)

    geo_download_examples.download_examples(["gse253003-counts"], repo=tmp_path)

    out_path = tmp_path / "data/experiments/raw/GSE253003/GSE253003_Count.csv.gz"
    assert out_path.is_file()
    assert out_path.read_bytes() == b"content:GSE253003_Count.csv.gz"
    assert seen_urls == [
        "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE253003&file=GSE253003_Count.csv.gz&format=file"
    ]


def test_download_examples_can_fetch_reference_bundle(tmp_path, monkeypatch):
    seen_urls = []

    def fake_get(url, stream, timeout):
        assert stream is True
        assert timeout == 120
        seen_urls.append(url)
        name = pathlib.Path(urlparse(url).path).name
        return FakeResponse([f"content:{name}".encode("utf-8")])

    monkeypatch.setattr(geo_download_examples.requests, "get", fake_get)

    geo_download_examples.download_examples(["ensembl-v109-refs"], repo=tmp_path)

    fasta_path = tmp_path / "data/experiments/raw/refs/ensembl_v109/Homo_sapiens.GRCh38.cdna.all.fa.gz"
    gtf_path = tmp_path / "data/experiments/raw/refs/ensembl_v109/Homo_sapiens.GRCh38.109.gtf.gz"
    assert fasta_path.is_file()
    assert gtf_path.is_file()
    assert fasta_path.read_bytes() == b"content:Homo_sapiens.GRCh38.cdna.all.fa.gz"
    assert gtf_path.read_bytes() == b"content:Homo_sapiens.GRCh38.109.gtf.gz"
    assert seen_urls == [
        "https://ftp.ensembl.org/pub/release-109/fasta/homo_sapiens/cdna/Homo_sapiens.GRCh38.cdna.all.fa.gz",
        "https://ftp.ensembl.org/pub/release-109/gtf/homo_sapiens/Homo_sapiens.GRCh38.109.gtf.gz",
    ]
