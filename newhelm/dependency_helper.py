from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import glob
import os
import shutil
import tempfile
from typing import Dict, Mapping, Optional

from newhelm.external_data import ExternalData
from newhelm.general import current_timestamp_millis, from_json, hash_file, to_json


class DependencyHelper(ABC):
    @abstractmethod
    def get_local_path(self, dependency_key: str) -> str:
        """Return a path to the dependency, downloading as needed."""

    @abstractmethod
    def versions_used(self) -> Mapping[str, str]:
        """Report the version of all dependencies accessed during this run."""

    @abstractmethod
    def update_all_dependencies(self) -> Mapping[str, str]:
        """Ensure the local system has the latest version of all dependencies."""


@dataclass(frozen=True)
class DependencyVersionMetadata:
    """Data object we can store along side a dependency version."""

    version: str
    creation_time_millis: int = field(default_factory=current_timestamp_millis)


class FromSourceDependencyHelper(DependencyHelper):
    """When a dependency isn't available locally, download from the primary source.

    When used, the local directory structure will look like this:
    data_dir/
      dependency_1/
        version_x.metadata
        version_x/
          <dependency's data>
        version_y.metadata
        version_y/
          <dependency's data>
        ...
      dependency_2/
        ...
      ...
    """

    def __init__(
        self,
        data_dir,
        dependencies: Mapping[str, ExternalData],
        required_versions: Mapping[str, str],
    ):
        self.data_dir = data_dir
        self.dependencies = dependencies
        self.required_versions = required_versions
        self.used_dependencies: Dict[str, str] = {}

    def get_local_path(self, dependency_key: str) -> str:
        assert dependency_key in self.dependencies
        external_data: ExternalData = self.dependencies[dependency_key]

        version: str
        if dependency_key in self.required_versions:
            version = self.required_versions[dependency_key]
            self._ensure_required_version_exists(dependency_key, external_data, version)
        else:
            version = self._get_latest_version(dependency_key, external_data)
        self.used_dependencies[dependency_key] = version
        return self._get_version_path(dependency_key, version)

    def versions_used(self) -> Mapping[str, str]:
        return self.used_dependencies

    def update_all_dependencies(self):
        latest_versions = {}
        for dependency_key, external_data in self.dependencies.items():
            latest_versions[dependency_key] = self._store_dependency(
                dependency_key, external_data
            )
        return latest_versions

    def _ensure_required_version_exists(
        self, dependency_key: str, external_data: ExternalData, version: str
    ) -> None:
        version_path = self._get_version_path(dependency_key, version)
        if os.path.exists(version_path):
            return
        # See if downloading from the source creates that version.
        stored_version = self._store_dependency(dependency_key, external_data)
        if stored_version != version:
            raise RuntimeError(
                f"Could not retrieve version {version} for dependency {dependency_key}. Source currently returns version {stored_version}."
            )

    def _get_latest_version(self, dependency_key, external_data) -> str:
        """Use the latest cached version. If none cached, download from source."""
        version = self._find_latest_cached_version(dependency_key)
        if version is not None:
            return version
        return self._store_dependency(dependency_key, external_data)

    def _get_version_path(self, dependency_key: str, version: str) -> str:
        # TODO Here or earlier, ensure dependency_key has no filesystem characters (e.g. '/').
        return os.path.join(self.data_dir, dependency_key, version)

    def _find_latest_cached_version(self, dependency_key: str) -> Optional[str]:
        # TODO Here or earlier, ensure dependency_key has no filesystem characters (e.g. '/').
        metadata_files = glob.glob(
            os.path.join(self.data_dir, dependency_key, "*.metadata")
        )
        version_creation: Dict[str, int] = {}
        for filename in metadata_files:
            with open(filename, "r") as f:
                metadata = from_json(DependencyVersionMetadata, f.read())
            version_creation[metadata.version] = metadata.creation_time_millis
        if not version_creation:
            return None
        # Returns the key with the max value
        return max(
            version_creation.keys(), key=lambda dict_key: version_creation[dict_key]
        )

    def _store_dependency(self, dependency_key, external_data: ExternalData) -> str:
        with tempfile.TemporaryDirectory() as tmpdirname:
            tmp_location = os.path.join(tmpdirname, dependency_key)
            external_data.download(tmp_location)
            version = hash_file(tmp_location)
            final_path = self._get_version_path(dependency_key, version)
            if os.path.exists(final_path):
                # TODO Allow for overwriting
                return version
            os.makedirs(os.path.join(self.data_dir, dependency_key), exist_ok=True)
            os.rename(tmp_location, final_path)
            # os.makedirs(final_path, exist_ok=True)
            # shutil.move(tmp_location, final_path)
            # TODO all the fancy unpacking in HELM's ensure_file_download.
            metadata_file = final_path + ".metadata"
            with open(metadata_file, "w") as f:
                f.write(to_json(DependencyVersionMetadata(version)))
            return version
