from pathlib import Path
from app.utils.filesystem import natural_sort_key, get_image_files

def test_natural_sort_key():
    files = ["image_10.jpg", "image_2.jpg", "image_1.jpg", "image_20.jpg"]
    sorted_files = sorted(files, key=natural_sort_key)
    assert sorted_files == ["image_1.jpg", "image_2.jpg", "image_10.jpg", "image_20.jpg"]

def test_get_image_files(synthetic_images):
    files = get_image_files(synthetic_images)
    
    # Verify .txt file is ignored
    assert not any(f.suffix == '.txt' for f in files)
    
    # Verify we get the expected number of images
    assert len(files) == 7  # 3 valid, 1 blurry, 1 over, 1 under, 1 low_res
    
    # Verify sorting
    names = [f.name for f in files]
    assert names[names.index("image_1.jpg")] < names[names.index("image_2.jpg")] < names[names.index("image_3.jpg")]
