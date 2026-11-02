import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from loguru import logger
import shutil

from app.config.settings import PreprocessingSettings
from app.utils.filesystem import get_image_files

class ImageProcessor:
    def __init__(self, settings: PreprocessingSettings):
        self.settings = settings
        
    def process_image(self, img: np.ndarray) -> np.ndarray:
        """Applies configured preprocessing steps to a single image."""
        if not self.settings.enabled:
            return img
            
        processed = img.copy()
        
        # Resize based on Target Megapixels
        if self.settings.resize.enabled:
            h, w = processed.shape[:2]
            current_mp = (h * w) / 1_000_000.0
            
            if current_mp > self.settings.target_mp:
                scale = np.sqrt(self.settings.target_mp / current_mp)
                new_w, new_h = int(w * scale), int(h * scale)
                processed = cv2.resize(processed, (new_w, new_h), interpolation=cv2.INTER_AREA)
                
        # Optional exposure correction (CLAHE) - Only if explicitly enabled
        if getattr(self.settings.exposure_correction, 'enabled', False) or self.settings.clahe.enabled:
            # Use CLAHE as a safe exposure normalization
            lab = cv2.cvtColor(processed, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            clahe = cv2.createCLAHE(
                clipLimit=self.settings.clahe.clip_limit, 
                tileGridSize=(self.settings.clahe.tile_grid_size, self.settings.clahe.tile_grid_size)
            )
            cl = clahe.apply(l)
            
            limg = cv2.merge((cl, a, b))
            processed = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
                
        # Denoise
        if self.settings.denoise.enabled:
            processed = cv2.fastNlMeansDenoisingColored(processed, None, 10, 10, 7, 21)
            
        # Optional Sharpening
        if getattr(self.settings.sharpen, 'enabled', False):
            # Very mild unsharp masking or Laplacian kernel
            kernel = np.array([[0, -1, 0], 
                               [-1, 5, -1], 
                               [0, -1, 0]])
            processed = cv2.filter2D(processed, -1, kernel)
            
        return processed

    def process_directory(self, input_dir: Path, output_dir: Path, quality_results: list[dict] = None) -> bool:
        """Processes images in input_dir and saves them to output_dir, skipping auto-rejected ones."""
        image_files = get_image_files(input_dir)
        if not image_files:
            return False
            
        # Identify rejected images if quality_results are provided
        rejected_filenames = set()
        if quality_results:
            rejected_filenames = {res["filename"] for res in quality_results if res.get("status") == "rejected"}
            
        output_dir.mkdir(parents=True, exist_ok=True)
        
        valid_files = [f for f in image_files if f.name not in rejected_filenames]
        logger.info(f"Preprocessing {len(valid_files)} images (skipped {len(image_files) - len(valid_files)} rejected)...")
        
        for img_path in tqdm(valid_files, desc="Preprocessing"):
            out_path = output_dir / img_path.name
            
            if self.settings.enabled:
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                    
                processed_img = self.process_image(img)
                cv2.imwrite(str(out_path), processed_img)
            else:
                # If disabled, just copy the file
                shutil.copy2(img_path, out_path)
                
        return True
