from newhelm.general import from_yaml
from newhelm.schema.test_config import (
    AdapterConfig,
    DatasetConfig,
    MetricConfig,
    ModelAnnotationConfig,
    PertubationConfig,
    StaticPromptResponseConfig,
    TestConfig,
)
from newhelm.test_utilities import parent_directory


def test_read_test_config_from_file(parent_directory):
    in_memory = TestConfig(
        name="Example Test",
        prompt_response_config=StaticPromptResponseConfig(
            dataset_config=DatasetConfig(
                dataset_name="best-data",
            ),
            perturbation_methods=[
                PertubationConfig(),
                PertubationConfig(),
            ],
            adapter_method=AdapterConfig(),
        ),
        model_annotation_config=[ModelAnnotationConfig()],
        metric_config=[
            MetricConfig(),
            MetricConfig(),
        ],
    )
    with parent_directory.joinpath("testdata", "test_config_example.yaml").open() as f:
        yaml_str = f.read()
    from_file = from_yaml(TestConfig, yaml_str)

    assert in_memory == from_file
