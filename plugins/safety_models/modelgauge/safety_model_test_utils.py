import pandas as pd
import numpy as np
from typing import List


def find_human_annotator_columns(dataset: pd.DataFrame) -> List[str]:
    """Search for human annotator columns indicated by the column name ending with _annotations"""
    return [column for column in dataset.columns if column.endswith("_annotations")]


def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the dataset. Replace nan values with None"""
    df = df.replace(np.nan, None)
    return df


def eval_dataset_validation(dataset: pd.DataFrame) -> bool:
    """Validate the dataset for expected columns"""
    is_valid = True

    print(f"Begin dataset validation")

    print(f"Checking for expected metadata columns")
    expected_metadata_columns = ["UID", "prompt", "response", "Hazard Category"]
    missing_columns = set(expected_metadata_columns) - set(dataset.columns)
    if missing_columns:
        print(f"Dataset is not valid. Expected columns: {missing_columns}")
    else:
        print("All expected metadata columns found")
        is_valid = False

    print(f"Checking for human annotater columns")
    human_annotator_columns = find_human_annotator_columns(dataset)
    if not human_annotator_columns:
        print("No human annotator columns found. Dataset is not valid.")
        is_valid = False
    else:
        for column in human_annotator_columns:
            print(f"Found human annotator column: {column}")

    # TODO type validation. Ensure that the columns are of the expected type
    # and the values are within the valid subset

    return is_valid
