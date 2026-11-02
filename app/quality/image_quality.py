import cv2
import numpy as np
from pathlib import Path
from typing import Any
from app.config.settings import QualitySettings
from app.utils.filesystem import get_image_files
from loguru import logger
import json

class ImageQualityAnalyzer:
    def __init__(self, settings: QualitySettings):
        self.settings = settings

    def analyze_image(self, image_path: Path) -> dict[str, Any]:
        """Analyzes a single image for quality metrics."""
        # Read image
        img = cv2.imread(str(image_path))
        if img is None:
            return {
                "filename": image_path.name,
                "status": "rejected",
                "reason": "unreadable"
            }
            
        height, width = img.shape[:2]
        
        # Check resolution
        if width < self.settings.min_width or height < self.settings.min_height:
            return {
                "filename": image_path.name,
                "width": width,
                "height": height,
                "status": "rejected",
                "reason": f"resolution too low ({width}x{height})"
            }
            
        # Convert to grayscale for metrics
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Blur score (Variance of Laplacian)
        blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        
        # Brightness (Mean)
        brightness = float(np.mean(gray))
        
        warnings = []
        
        # Blur Classification
        if blur_score < self.settings.blur_threshold:
            blur_class = "BLURRY"
            warnings.append("BLURRY")
        elif blur_score < self.settings.blur_threshold * 1.5:
            blur_class = "ACCEPTABLE"
        else:
            blur_class = "SHARP"
            
        # Exposure Classification
        if brightness < self.settings.brightness_min:
            exposure_class = "UNDEREXPOSED"
            warnings.append("UNDEREXPOSED")
        elif brightness > self.settings.brightness_max:
            exposure_class = "OVEREXPOSED"
            warnings.append("OVEREXPOSED")
        else:
            exposure_class = "NORMAL"
            
        # Determine Status
        status = "accepted"
        if warnings:
            status = "warning"
            if "BLURRY" in warnings and self.settings.auto_reject_blurry:
                status = "rejected"
            
        return {
            "filename": image_path.name,
            "width": width,
            "height": height,
            "blur_score": blur_score,
            "blur_class": blur_class,
            "brightness": brightness,
            "exposure_class": exposure_class,
            "status": status,
            "warnings": warnings,
            "reason": ", ".join(warnings) if status == "rejected" else ""
        }

    def process_directory(self, input_dir: Path) -> list[dict[str, Any]]:
        """
        Analyzes all images in a directory and returns a list of result dictionaries.
        """
        image_files = get_image_files(input_dir)
        total_images = len(image_files)
        
        if total_images == 0:
            logger.error(f"No valid images found in {input_dir}")
            return []
            
        if total_images < self.settings.min_images:
            logger.warning(f"Only {total_images} images found. At least {self.settings.min_images} are recommended.")
            
        results = []
        
        for img_path in image_files:
            res = self.analyze_image(img_path)
            results.append(res)
            
            for w in res.get("warnings", []):
                logger.warning(f"Image {img_path.name} appears {w}")
                
        return results
