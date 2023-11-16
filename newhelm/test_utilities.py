import pathlib
import pytest


@pytest.fixture
def parent_directory(request):
    file = pathlib.Path(request.node.fspath)
    return file.parent
