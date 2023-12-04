from dataclasses import dataclass
import os
from typing import Dict, List, Mapping
from newhelm.example_importers.bbq_example_importer import BBQExampleImporter

from newhelm.example_importers.example_importer import ExampleImporter
from newhelm.dependency_helper import FromSourceDependencyHelper
from newhelm.instance import Instance


@dataclass(frozen=True)
class VersionedExamples:
    """Container to make demo outputs more clear."""

    importer_name: str
    versions_used: Mapping[str, str]
    examples: List[Instance]


def get_versioned_examples(
    importer: ExampleImporter,
    desired_subsets,
    output_dir: str,
    version_map: Mapping[str, Mapping[str, str]],
) -> VersionedExamples:
    """This method isn't meant to live long term, just a demostration for how this could all fit together."""
    importer_name = importer.get_metadata().name
    data_dir = os.path.join(output_dir, importer_name, "data")
    try:
        required_versions = version_map[importer_name]
    except KeyError:
        required_versions = {}

    dependency_helper = FromSourceDependencyHelper(
        data_dir,
        dependencies=importer.get_dependencies(),
        required_versions=required_versions,
    )
    examples = importer.get_examples(dependency_helper, subsets=desired_subsets)
    versions_used = dependency_helper.versions_used()
    return VersionedExamples(
        importer_name=importer_name,
        versions_used=versions_used,
        examples=examples,
    )


# Pick some arguments just to make it do things
importer = BBQExampleImporter()
desired_subsets = ["Age"]
output_dir = "benchmark_output/scenarios/"
version_map: Mapping[str, Mapping[str, str]] = {
    "bbq": {"Age": "46e805b3fc2d8cbd26eeb8e8430d98cf7b2dc9c83574ff3674e8ce4f0fca2a60"}
}

versioned_examples = get_versioned_examples(
    importer, desired_subsets, output_dir, version_map
)
print(versioned_examples.importer_name)
print(versioned_examples.versions_used)
print(len(versioned_examples.examples))
