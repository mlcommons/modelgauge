from collections import namedtuple
from unittest.mock import ANY
import pytest
from newhelm.external_data import WebData, GDriveData, LocalData

GDriveFileToDownload = namedtuple("GDriveFileToDownload", ("id", "path"))


def test_web_data_download(mocker):
    mock_download = mocker.patch("urllib.request.urlretrieve")
    web_data = WebData(source_url="http://example.com")
    web_data.download("test.tgz")
    mock_download.assert_called_once_with(
        "http://example.com", "test.tgz", reporthook=ANY
    )


def test_gdrive_data_download(mocker):
    mock_download_folder = mocker.patch(
        "gdown.download_folder",
        return_value=[GDriveFileToDownload("file_id", "file.csv")],
    )
    mock_download_file = mocker.patch("gdown.download")
    gdrive_data = GDriveData(folder_url="http://example_drive.com", filename="file.csv")
    gdrive_data.download("test.tgz")
    mock_download_folder.assert_called_once_with(
        url="http://example_drive.com", skip_download=True, quiet=True
    )
    mock_download_file.assert_called_once_with(id="file_id", output="test.tgz")


def test_gdrive_correct_file_download(mocker):
    """Checks that correct file is downloaded if multiple files exist in the folder."""
    mock_download_folder = mocker.patch(
        "gdown.download_folder",
        return_value=[
            GDriveFileToDownload("file_id1", "different_file.csv"),
            GDriveFileToDownload("file_id2", "file.txt"),
            GDriveFileToDownload("file_id3", "file.csv"),
        ],
    )
    mock_download_file = mocker.patch("gdown.download")
    gdrive_data = GDriveData(folder_url="http://example_drive.com", filename="file.csv")
    gdrive_data.download("test.tgz")
    mock_download_folder.assert_called_once_with(
        url="http://example_drive.com", skip_download=True, quiet=True
    )
    mock_download_file.assert_called_once_with(id="file_id3", output="test.tgz")


def test_gdrive_nonexistent_filename(mocker):
    """Throws exception when the folder does not contain any files with the desired filename."""
    mock_download_folder = mocker.patch(
        "gdown.download_folder",
        return_value=[
            GDriveFileToDownload("file_id1", "different_file.csv"),
            GDriveFileToDownload("file_id2", "file.txt"),
        ],
    )
    mock_download_file = mocker.patch("gdown.download")
    gdrive_data = GDriveData(folder_url="http://example_drive.com", filename="file.csv")
    with pytest.raises(Exception, match="Cannot find file"):
        gdrive_data.download("test.tgz")


def test_local_data_download(mocker):
    mock_copy = mocker.patch("shutil.copy")
    local_data = LocalData(path="origin_test.tgz")
    local_data.download("destintation_test.tgz")
    mock_copy.assert_called_once_with("origin_test.tgz", "destintation_test.tgz")
