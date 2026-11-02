import pytest
import sqlite3
import numpy as np
from pathlib import Path
from app.reconstruction.colmap_analyzer import ColmapAnalyzer

@pytest.fixture
def mock_colmap_db(tmp_path):
    db_path = tmp_path / "mock_database.db"
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    
    c.execute("CREATE TABLE images (image_id INTEGER PRIMARY KEY, name TEXT);")
    c.execute("INSERT INTO images VALUES (1, '00001.jpg'), (2, '00002.jpg'), (3, '00003.jpg');")
    
    c.execute("CREATE TABLE keypoints (image_id INTEGER, rows INTEGER, cols INTEGER, data BLOB);")
    # rows is populated. image 1: 50, image 2: 100, image 3: 150
    # data can be empty bytes for now since rows > 0
    c.execute("INSERT INTO keypoints VALUES (1, 50, 6, ?), (2, 100, 6, ?), (3, 150, 6, ?)", (b'', b'', b''))
    
    c.execute("CREATE TABLE matches (pair_id INTEGER, rows INTEGER, cols INTEGER, data BLOB);")
    # pair_id logic: 1-2 => 1 * 2147483647 + 2 = 2147483649
    # pair 1-2: 30 raw matches
    c.execute("INSERT INTO matches VALUES (2147483649, 30, 2, ?)", (b'',))
    
    c.execute("CREATE TABLE two_view_geometries (pair_id INTEGER, rows INTEGER, cols INTEGER, data BLOB);")
    # pair 1-2: 20 verified matches
    c.execute("INSERT INTO two_view_geometries VALUES (2147483649, 20, 2, ?)", (b'',))
    
    conn.commit()
    conn.close()
    return db_path

def test_colmap_analyzer_counts(mock_colmap_db):
    analyzer = ColmapAnalyzer(mock_colmap_db)
    
    features = analyzer.analyze_features()
    assert features["00001.jpg"] == 50
    assert features["00002.jpg"] == 100
    assert features["00003.jpg"] == 150
    
    # Check fallback logic using BLOB directly without rows
    conn = sqlite3.connect(mock_colmap_db)
    c = conn.cursor()
    c.execute("UPDATE keypoints SET rows = 0 WHERE image_id = 1")
    # if rows = 0, fallback to data size. 10 float32 points * 6 cols * 4 bytes = 240 bytes
    c.execute("UPDATE keypoints SET data = ? WHERE image_id = 1", (b'x' * 240,))
    
    c.execute("UPDATE matches SET rows = 0")
    # 25 matches * 2 cols * 4 bytes = 200 bytes
    c.execute("UPDATE matches SET data = ?", (b'x' * 200,))
    
    c.execute("UPDATE two_view_geometries SET rows = 0")
    # 15 matches * 2 cols * 4 bytes = 120 bytes
    c.execute("UPDATE two_view_geometries SET data = ?", (b'x' * 120,))
    
    conn.commit()
    conn.close()
    
    features2 = analyzer.analyze_features()
    # 240 / (6 * 4) = 10
    assert features2["00001.jpg"] == 10
    
    matches = analyzer.analyze_matches(features2, min_matches=10, min_match_ratio=0.01)
    
    pairs = matches["pairs"]
    assert len(pairs) == 1
    p = pairs[0]
    
    # 120 / (2 * 4) = 15 verified
    assert p["match_count"] == 15
    # 200 / (2 * 4) = 25 raw
    assert p["raw_matches"] == 25
