import pytest
from unittest.mock import patch
from pathlib import Path
from app.reconstruction.colmap_runner import ColmapRunner

def test_colmap_runner_resolution_absolute_path_exists(tmp_path):
    # Create a dummy executable
    dummy_exe = tmp_path / "colmap.exe"
    dummy_exe.touch()
    
    runner = ColmapRunner(colmap_path=str(dummy_exe))
    assert runner.colmap_path == str(dummy_exe)

def test_colmap_runner_resolution_absolute_path_missing(tmp_path):
    dummy_exe = tmp_path / "missing.exe"
    
    with pytest.raises(FileNotFoundError, match="COLMAP executable not found"):
        ColmapRunner(colmap_path=str(dummy_exe))

@patch("app.reconstruction.colmap_runner.shutil.which")
def test_colmap_runner_resolution_shutil_which_success(mock_which):
    mock_which.return_value = "/mock/path/to/colmap"
    
    runner = ColmapRunner(colmap_path="colmap")
    assert runner.colmap_path == "/mock/path/to/colmap"
    mock_which.assert_called_once_with("colmap")

@patch("app.reconstruction.colmap_runner.shutil.which")
def test_colmap_runner_resolution_shutil_which_failure(mock_which):
    mock_which.return_value = None
    
    with pytest.raises(FileNotFoundError, match="COLMAP executable not found"):
        ColmapRunner(colmap_path="colmap")
