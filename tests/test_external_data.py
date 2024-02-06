import pytest

from newhelm.external_data import WebData, LocalData


@pytest.fixture
def mock_download(mocker):
    return mocker.patch("urllib.request.urlretrieve")


@pytest.fixture
def mock_copy(mocker):
    return mocker.patch("shutil.copy")


def test_web_data_download(mock_download):
    web_data = WebData(source_url="http://example.com")
    web_data.download("test.tgz")
    mock_download.assert_called_once_with("http://example.com", "test.tgz")


def test_local_data_download(mock_copy):
    local_data = LocalData(path="test.tgz")
    local_data.download("test.tgz")
    mock_copy.assert_called_once_with("test.tgz", "test.tgz")
