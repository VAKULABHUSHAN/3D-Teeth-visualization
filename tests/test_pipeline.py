import pytest
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock
import sqlite3

from app.reconstruction.reconstruction_pipeline import ReconstructionPipeline, get_database_statistics
from app.config.settings import ReconstructionSettings, MatchingSettings
from app.utils.filesystem import get_image_files

def test_database_statistics_empty(tmp_path):
    db_path = tmp_path / "missing.db"
    stats = get_database_statistics(db_path)
    assert stats["images"] == 0
    assert stats["matches"] == 0

def test_database_statistics_valid(tmp_path):
    db_path = tmp_path / "valid.db"
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("CREATE TABLE images (image_id INTEGER PRIMARY KEY);")
    c.execute("INSERT INTO images VALUES (1), (2), (3);")
    c.execute("CREATE TABLE cameras (camera_id INTEGER PRIMARY KEY);")
    c.execute("CREATE TABLE keypoints (image_id INTEGER);")
    c.execute("CREATE TABLE matches (pair_id INTEGER);")
    c.execute("CREATE TABLE two_view_geometries (pair_id INTEGER);")
    conn.commit()
    conn.close()
    
    stats = get_database_statistics(db_path)
    assert stats["images"] == 3
    assert stats["cameras"] == 0

@patch("app.reconstruction.reconstruction_pipeline.ColmapRunner")
def test_reconstruction_pipeline_fails_on_missing_images(MockRunner, tmp_path):
    recon_settings = ReconstructionSettings(colmap_path="colmap", camera_model="SIMPLE_RADIAL", single_camera=True)
    match_settings = MatchingSettings(strategy="exhaustive")
    
    pipeline = ReconstructionPipeline(recon_settings, match_settings)
    mock_runner_instance = MockRunner.return_value
    mock_runner_instance.feature_extractor.return_value = False
    
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    (image_dir / "001.jpg").touch()
    
    db_path = tmp_path / "output" / "database.db"
    
    success = pipeline.run_extraction(image_dir, db_path)
    assert success is False
    mock_runner_instance.feature_extractor.assert_called_once()

@patch("app.reconstruction.reconstruction_pipeline.ColmapRunner")
def test_reconstruction_pipeline_fails_on_zero_matches(MockRunner, tmp_path):
    recon_settings = ReconstructionSettings(colmap_path="colmap", camera_model="SIMPLE_RADIAL", single_camera=True)
    match_settings = MatchingSettings(strategy="exhaustive")
    
    pipeline = ReconstructionPipeline(recon_settings, match_settings)
    mock_runner_instance = MockRunner.return_value
    mock_runner_instance.exhaustive_matcher.return_value = False
    
    db_path = tmp_path / "output" / "database.db"
    
    success = pipeline.run_matching(db_path)
    assert success is False
    mock_runner_instance.exhaustive_matcher.assert_called_once()

@patch("app.reconstruction.reconstruction_pipeline.ColmapRunner")
def test_reconstruction_pipeline_paths_consistent(MockRunner, tmp_path):
    recon_settings = ReconstructionSettings(colmap_path="colmap", camera_model="SIMPLE_RADIAL", single_camera=True)
    match_settings = MatchingSettings(strategy="exhaustive")
    
    pipeline = ReconstructionPipeline(recon_settings, match_settings)
    mock_runner_instance = MockRunner.return_value
    mock_runner_instance.feature_extractor.return_value = True
    mock_runner_instance.exhaustive_matcher.return_value = True
    mock_runner_instance.mapper.return_value = True
    mock_runner_instance.model_converter.return_value = True
    
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    (image_dir / "001.jpg").touch()
    
    output_dir = tmp_path / "output"
    db_path = output_dir / "database.db"
    
    with patch("app.reconstruction.reconstruction_pipeline.get_database_statistics") as mock_stats:
        mock_stats.return_value = {"cameras": 1, "images": 1, "keypoints": 100, "matches": 10, "two_view_geometries": 5}
        
        # Create fake output for mapper
        def mock_mapper(*args, **kwargs):
            sparse_dir = kwargs['output_path']
            model_0 = sparse_dir / "0"
            model_0.mkdir(parents=True)
            (model_0 / "points3D.bin").touch()
            return True
            
        mock_runner_instance.mapper.side_effect = mock_mapper
        
        # Run stages separately
        pipeline.run_extraction(image_dir, db_path)
        pipeline.run_matching(db_path)
        res = pipeline.run_mapper(image_dir, db_path, output_dir)
        
        assert res["success"] is True
        
        mock_runner_instance.feature_extractor.assert_called_with(
            database_path=db_path,
            image_path=image_dir.resolve(),
            camera_model="SIMPLE_RADIAL",
            single_camera=True
        )
        mock_runner_instance.exhaustive_matcher.assert_called_with(
            database_path=db_path
        )
        mock_runner_instance.mapper.assert_called_with(
            database_path=db_path,
            image_path=image_dir.resolve(),
            output_path=output_dir.resolve() / "sparse",
            mapper_options=None
        )
