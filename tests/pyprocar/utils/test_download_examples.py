import fnmatch
import tempfile
import zipfile
from pathlib import Path

import pytest

import pyprocar.utils.download_examples as download_examples

DATASET = {"data/codes.zip": "codes", "data/codes-extra.zip": "extra"}


@pytest.fixture
def hub(tmp_path, monkeypatch):
    downloaded: list[str] = []
    staged: list[str] = []
    shared_cache = tmp_path / "hub" / "datasets--fake"
    (shared_cache / "refs").mkdir(parents=True)
    (shared_cache / "refs" / "main").write_text("rev")

    def snapshot_download(*, allow_patterns, local_dir=None, **_hub_args):
        if local_dir:
            staged.append(local_dir)
        root = Path(local_dir) if local_dir else shared_cache / "snapshots" / "rev"
        root.mkdir(parents=True, exist_ok=True)
        for name, body in DATASET.items():
            if any(fnmatch.fnmatch(name, p) for p in allow_patterns):
                downloaded.append(name)
                (root / name).parent.mkdir(parents=True, exist_ok=True)
                with zipfile.ZipFile(root / name, "w") as z:
                    z.writestr("qe/scf.out", body)
        return str(root)

    monkeypatch.setattr(download_examples, "snapshot_download", snapshot_download)
    out = tmp_path / "out"
    (out / ".cache").mkdir(parents=True)
    (out / ".cache" / "keep").write_text("mine")
    return out, downloaded, shared_cache, staged


def test_download_from_hf_fetches_only_the_archive_named_exactly(hub):
    out, downloaded, shared_cache, _ = hub

    download_examples.download_from_hf("data/codes", output_path=out)

    assert downloaded == ["data/codes.zip"]
    assert (out / "data/codes/qe/scf.out").read_text() == "codes"
    assert sorted(p.name for p in (out / "data").iterdir()) == ["codes"]
    assert (out / ".cache" / "keep").read_text() == "mine"
    assert (shared_cache / "refs" / "main").read_text() == "rev"


def test_download_from_hf_refuses_a_prefix_without_downloading(hub):
    out, downloaded, _, _ = hub

    with pytest.raises(FileNotFoundError) as refused:
        download_examples.download_from_hf("data/c", output_path=out)

    assert downloaded == []
    assert str(refused.value) == "data/c.zip is not in the lllangWV/pyprocar_test_data dataset"
    assert [p.name for p in (out / "data").iterdir()] == []


def test_download_from_hf_stages_beside_the_fixture_not_in_the_system_tmp(
    hub, tmp_path, monkeypatch
):
    out, _, _, staged = hub
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path / "no-system-tmp"))

    download_examples.download_from_hf("data/codes", output_path=out)

    assert (out / "data/codes/qe/scf.out").read_text() == "codes"
    assert [Path(s).parent for s in staged] == [out / "data"]
    assert sorted(p.name for p in (out / "data").iterdir()) == ["codes"]


def test_download_from_hf_unpacks_a_whole_directory_archive_in_place(tmp_path, monkeypatch):
    def snapshot_download(*, allow_patterns, local_dir, **_hub_args):
        archive = Path(local_dir) / allow_patterns[0]
        archive.parent.mkdir(parents=True)
        # every entry of the dataset's data/io.zip starts with io/, as compress_dirpath writes it
        with zipfile.ZipFile(archive, "w") as z:
            z.writestr("io/vasp/outcar/OUTCAR_v64", "outcar")

    monkeypatch.setattr(download_examples, "snapshot_download", snapshot_download)

    download_examples.download_from_hf("data/io", output_path=tmp_path)

    assert (tmp_path / "data/io/vasp/outcar/OUTCAR_v64").read_text() == "outcar"
    assert sorted(p.name for p in (tmp_path / "data/io").iterdir()) == ["vasp"]


def test_a_compressed_directory_unpacks_to_the_same_tree(tmp_path):
    source = tmp_path / "codes"
    (source / "qe").mkdir(parents=True)
    (source / "qe" / "scf.out").write_text("scf")
    download_examples.compress_dirpath(source)
    (source / "qe" / "scf.out").unlink()
    (source / "qe").rmdir()
    source.rmdir()

    download_examples.uncompress_dirpath(tmp_path / "codes.zip")

    assert (tmp_path / "codes/qe/scf.out").read_text() == "scf"
    assert sorted(p.name for p in (tmp_path / "codes").iterdir()) == ["qe"]
