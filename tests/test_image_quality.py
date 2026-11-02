import json
from app.quality.image_quality import ImageQualityAnalyzer
from app.config.settings import QualitySettings

def test_image_quality_analysis(synthetic_images, tmp_path):
    settings = QualitySettings(
        min_images=3,
        min_width=640,
        min_height=480,
        blur_threshold=10.0,
        brightness_min=20.0,
        brightness_max=240.0,
        auto_reject_blurry=False,
        auto_reject_low_feature=False
    )
    
    analyzer = ImageQualityAnalyzer(settings)
    
    # Test valid image
    res_valid = analyzer.analyze_image(synthetic_images / "image_2.jpg")
    assert res_valid["status"] == "accepted"
    assert len(res_valid["warnings"]) == 0
    
    # Test low resolution
    res_low = analyzer.analyze_image(synthetic_images / "low_res.jpg")
    assert res_low["status"] == "rejected"
    assert "resolution too low" in res_low["reason"]
    
    # Test blurry
    res_blurry = analyzer.analyze_image(synthetic_images / "blurry.jpg")
    assert "BLURRY" in res_blurry["warnings"]
    assert res_blurry["status"] == "warning" # Because auto_reject_blurry is False
    
    # Test underexposed
    res_under = analyzer.analyze_image(synthetic_images / "under.jpg")
    assert "UNDEREXPOSED" in res_under["warnings"]
    
    # Test overexposed
    res_over = analyzer.analyze_image(synthetic_images / "over.jpg")
    assert "OVEREXPOSED" in res_over["warnings"]
    
    # Test process directory
    results = analyzer.process_directory(synthetic_images)
    
    assert len(results) == 7
    # Count how many have status != rejected (only low_res is rejected by resolution)
    valid = [r for r in results if r["status"] != "rejected"]
    assert len(valid) == 6
