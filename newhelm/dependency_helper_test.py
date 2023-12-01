import time

import pytest
from newhelm.dependency_helper import (
    DependencyVersionMetadata,
    FromSourceDependencyHelper,
)
from newhelm.external_data import ExternalData
from newhelm.general import from_json


class MockExternalData(ExternalData):
    def __init__(self, text):
        self.download_calls = 0
        self.text = text

    def download(self, location):
        self.download_calls += 1
        with open(location, "w") as f:
            f.write(self.text)


def test_from_source_single_read(tmpdir):
    dependencies = {
        "d1": MockExternalData("data-1"),
        "d2": MockExternalData("data-2"),
    }
    # This is the hash for "data-1"
    d1_hash = "51bbfa74f8660493f40fd72068f63af436ee13c283ca84c373d9690ff2f1f83c"
    helper = FromSourceDependencyHelper(
        tmpdir.strpath, dependencies, required_versions={}
    )

    # Get the d1 dependency
    d1_path = helper.get_local_path("d1")

    assert d1_path.endswith(f"d1/{d1_hash}")
    assert helper.versions_used() == {"d1": d1_hash}
    assert dependencies["d1"].download_calls == 1
    assert dependencies["d2"].download_calls == 0

    # Ensure the file contains the expected data.
    with open(d1_path, "r") as f:
        d1_from_file = f.read()
    assert d1_from_file == "data-1"

    # Ensure the .metadata file was written
    with open(d1_path + ".metadata", "r") as f:
        metadata = from_json(DependencyVersionMetadata, f.read())
    assert metadata.version == d1_hash
    assert metadata.creation_time_millis > 0


def test_from_source_required_version_already_exists(tmpdir):
    # Make the old version
    old_dependencies = {
        "d1": MockExternalData("data-1"),
    }
    # This is the hash for "data-1"
    d1_hash = "51bbfa74f8660493f40fd72068f63af436ee13c283ca84c373d9690ff2f1f83c"
    old_helper = FromSourceDependencyHelper(
        tmpdir.strpath, old_dependencies, required_versions={}
    )

    # Get the d1 dependency
    old_d1_path = old_helper.get_local_path("d1")

    new_dependencies = {
        "d1": MockExternalData("updated-data-1"),
    }
    new_helper = FromSourceDependencyHelper(
        tmpdir.strpath, new_dependencies, required_versions={"d1": d1_hash}
    )
    new_d1_path = new_helper.get_local_path("d1")
    assert old_d1_path == new_d1_path
    # Read it from storage.
    assert new_dependencies["d1"].download_calls == 0


def test_from_source_required_version_is_live(tmpdir):
    dependencies = {
        "d1": MockExternalData("data-1"),
    }
    # This is the hash for "data-1"
    d1_hash = "51bbfa74f8660493f40fd72068f63af436ee13c283ca84c373d9690ff2f1f83c"
    helper = FromSourceDependencyHelper(
        tmpdir.strpath, dependencies, required_versions={"d1": d1_hash}
    )

    # Get the d1 dependency
    d1_path = helper.get_local_path("d1")
    assert d1_path.endswith(d1_hash)


def test_from_source_required_version_unavailable(tmpdir):
    dependencies = {
        "d1": MockExternalData("data-1"),
    }
    helper = FromSourceDependencyHelper(
        tmpdir.strpath, dependencies, required_versions={"d1": "not-real"}
    )

    with pytest.raises(RuntimeError, match="version not-real for dependency d1"):
        helper.get_local_path("d1")


def test_from_source_require_older_version(tmpdir):
    # First write a version of 'd1' with contents of 'data-1'.
    old_dependencies = {
        "d1": MockExternalData("data-1"),
    }
    # This is the hash for "data-1"
    d1_hash = "51bbfa74f8660493f40fd72068f63af436ee13c283ca84c373d9690ff2f1f83c"
    old_helper = FromSourceDependencyHelper(
        tmpdir.strpath, old_dependencies, required_versions={}
    )
    old_d1_path = old_helper.get_local_path("d1")
    time.sleep(0.05)  # Ensure timestamp of old is actually older.

    # Now write a newer version of d1
    new_dependencies = {
        "d1": MockExternalData("updated-data-1"),
    }
    new_helper = FromSourceDependencyHelper(
        tmpdir.strpath, new_dependencies, required_versions={}
    )
    # Force reading the new data.
    new_helper.update_all_dependencies()
    new_d1_path = new_helper.get_local_path("d1")
    assert old_d1_path != new_d1_path

    # Finally, set up a helper with a required version.
    required_version_helper = FromSourceDependencyHelper(
        tmpdir.strpath, new_dependencies, required_versions={"d1": d1_hash}
    )
    required_version_d1_path = required_version_helper.get_local_path("d1")
    assert new_d1_path != required_version_d1_path
    with open(new_d1_path, "r") as f:
        assert f.read() == "updated-data-1"
    with open(required_version_d1_path, "r") as f:
        assert f.read() == "data-1"


def test_from_source_reads_cached(tmpdir):
    dependencies = {
        "d1": MockExternalData("data-1"),
        "d2": MockExternalData("data-2"),
    }

    helper = FromSourceDependencyHelper(
        tmpdir.strpath, dependencies, required_versions={}
    )
    d1_path = helper.get_local_path("d1")
    d1_path_again = helper.get_local_path("d1")
    assert d1_path == d1_path_again
    assert dependencies["d1"].download_calls == 1


def test_from_source_update_all(tmpdir):
    dependencies = {
        "d1": MockExternalData("data-1"),
        "d2": MockExternalData("data-2"),
    }
    helper = FromSourceDependencyHelper(
        tmpdir.strpath, dependencies, required_versions={}
    )
    versions = helper.update_all_dependencies()
    assert versions == {
        "d1": "51bbfa74f8660493f40fd72068f63af436ee13c283ca84c373d9690ff2f1f83c",
        "d2": "00c2022f72beeabc82c8f02099df7abebe43292bac3f44bf63f5827a8c50255a",
    }
    assert dependencies["d1"].download_calls == 1
    assert dependencies["d2"].download_calls == 1
    # Nothing has actually been read.
    assert helper.versions_used() == {}
    # TODO read files?
