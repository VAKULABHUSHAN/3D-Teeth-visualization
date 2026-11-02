import pytest
import cv2
import numpy as np
from pathlib import Path

@pytest.fixture
def synthetic_images(tmp_path):
    """Generates synthetic images for testing."""
    test_dir = tmp_path / "test_images"
    test_dir.mkdir()
    
    # Generate valid images
    for i in range(1, 4):
        # Create a simple valid image (gradient)
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        img[:] = (i * 50, i * 50, i * 50)
        # Add some edges to not be totally blurry
        cv2.rectangle(img, (100, 100), (500, 300), (255, 255, 255), 2)
        cv2.imwrite(str(test_dir / f"image_{i}.jpg"), img)
        
    # Generate a blurry image
    blurry = np.ones((480, 640, 3), dtype=np.uint8) * 128
    cv2.imwrite(str(test_dir / "blurry.jpg"), blurry)
    
    # Generate an overexposed image
    overexposed = np.ones((480, 640, 3), dtype=np.uint8) * 255
    cv2.imwrite(str(test_dir / "over.jpg"), overexposed)
    
    # Generate an underexposed image
    underexposed = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.imwrite(str(test_dir / "under.jpg"), underexposed)
    
    # Generate a low res image
    low_res = np.zeros((240, 320, 3), dtype=np.uint8)
    cv2.imwrite(str(test_dir / "low_res.jpg"), low_res)
    
    # Unsupported file
    (test_dir / "unsupported.txt").write_text("not an image")
    
    return test_dir
