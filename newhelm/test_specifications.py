from typing import Any, Generator, Mapping, Optional
from pydantic import BaseModel
import tomli
import newhelm.tests.specifications
from importlib import resources


class Identity(BaseModel):
    uid: str
    version: Optional[str] = None
    display_name: str


class TestSpecification(BaseModel):
    source: str
    """Source is NOT in the toml file.
    
    For toml files, this is the path the file was loaded from.
    """
    identity: Identity

    # TODO The rest of the fields.


def load_module_toml_files(module) -> Generator[dict[str, Any]]:
    for path in resources.files(module).iterdir():
        if not path.is_file():
            continue
        if not path.name.endswith(".toml"):
            continue
        try:
            with path.open("rb") as f:
                yield tomli.load(f)
        except Exception as e:
            raise Exception(f"While processing {path}.") from e


def load_test_specification_files() -> Mapping[str, TestSpecification]:
    results = {}
    for path in resources.files(newhelm.tests.specifications).iterdir():
        if not path.is_file():
            continue
        if not path.name.endswith(".toml"):
            continue
        try:
            with path.open("rb") as f:
                raw = tomli.load(f)
            assert "source" not in raw
            raw["source"] = str(path)
            parsed = TestSpecification.model_validate(raw, strict=True)
        except Exception as e:
            raise Exception(f"While processing {path}.") from e
        uid = parsed.identity.uid
        assert uid not in results
        results[uid] = parsed
    return results
