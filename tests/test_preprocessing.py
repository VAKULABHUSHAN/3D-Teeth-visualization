import cv2
from app.preprocessing.image_processor import ImageProcessor
from app.config.settings import PreprocessingSettings, PreprocessingResizeSettings, PreprocessingDenoiseSettings, PreprocessingClaheSettings, PreprocessingSharpenSettings, PreprocessingExposureSettings
from app.utils.filesystem import get_image_files

def test_preprocessing_pipeline(synthetic_images, tmp_path):
    settings = PreprocessingSettings(
        enabled=True,
        target_mp=0.1,  # 100k pixels
        resize=PreprocessingResizeSettings(enabled=True),
        denoise=PreprocessingDenoiseSettings(enabled=True),
        clahe=PreprocessingClaheSettings(enabled=True, clip_limit=2.0, tile_grid_size=8),
        sharpen=PreprocessingSharpenSettings(enabled=False),
        exposure_correction=PreprocessingExposureSettings(enabled=False)
    )
    
    processor = ImageProcessor(settings)
    output_dir = tmp_path / "processed"
    
    success = processor.process_directory(synthetic_images, output_dir)
    assert success
    
    # Check outputs
    processed_files = get_image_files(output_dir)
    assert len(processed_files) == 7
    
    # Check if resize worked
    for p in processed_files:
        img = cv2.imread(str(p))
        h, w = img.shape[:2]
        # Allow small floating point rounding differences
        assert (h * w) / 1_000_000.0 <= 0.11

def test_preprocessing_disabled(synthetic_images, tmp_path):
    settings = PreprocessingSettings(
        enabled=False,
        target_mp=0.1,
        resize=PreprocessingResizeSettings(enabled=True),
        denoise=PreprocessingDenoiseSettings(enabled=True),
        clahe=PreprocessingClaheSettings(enabled=True, clip_limit=2.0, tile_grid_size=8),
        sharpen=PreprocessingSharpenSettings(enabled=False),
        exposure_correction=PreprocessingExposureSettings(enabled=False)
    )
    
    processor = ImageProcessor(settings)
    output_dir = tmp_path / "processed_disabled"
    
    success = processor.process_directory(synthetic_images, output_dir)
    assert success
    
    # Resize should not have happened
    img = cv2.imread(str(output_dir / "image_1.jpg"))
    h, w = img.shape[:2]
    assert w == 640 and h == 480
