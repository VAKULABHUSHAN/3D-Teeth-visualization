import json
import time
from pathlib import Path
from loguru import logger
import shutil
import sqlite3

from app.config.settings import ReconstructionSettings, MatchingSettings
from app.reconstruction.colmap_runner import ColmapRunner
from app.utils.filesystem import get_image_files

def get_database_statistics(database_path: Path) -> dict:
    if not database_path.exists():
        return {"cameras": 0, "images": 0, "keypoints": 0, "matches": 0, "two_view_geometries": 0}
        
    try:
        conn = sqlite3.connect(str(database_path))
        c = conn.cursor()
        
        def count_table(table):
            try:
                c.execute(f"SELECT COUNT(*) FROM {table}")
                return c.fetchone()[0]
            except sqlite3.OperationalError:
                return 0
                
        stats = {
            "cameras": count_table("cameras"),
            "images": count_table("images"),
            "keypoints": count_table("keypoints"),
            "matches": count_table("matches"),
            "two_view_geometries": count_table("two_view_geometries")
        }
        conn.close()
        return stats
    except Exception as e:
        logger.error(f"Failed to read database statistics: {e}")
        return {"cameras": 0, "images": 0, "keypoints": 0, "matches": 0, "two_view_geometries": 0}

class ReconstructionPipeline:
    def __init__(self, recon_settings: ReconstructionSettings, match_settings: MatchingSettings):
        self.recon_settings = recon_settings
        self.match_settings = match_settings
        self.runner = ColmapRunner(colmap_path=recon_settings.colmap_path)

    def run_extraction(self, image_dir: Path, database_path: Path) -> bool:
        """Runs COLMAP feature extraction."""
        logger.info("Running feature extraction...")
        
        image_files = get_image_files(image_dir)
        input_images = len(image_files)
        
        success = self.runner.feature_extractor(
            database_path=database_path,
            image_path=image_dir,
            camera_model=self.recon_settings.camera_model,
            single_camera=self.recon_settings.single_camera
        )
        if not success:
            logger.error("Feature extraction failed.")
            return False
            
        stats = get_database_statistics(database_path)
        logger.info(f"Database after extraction: {stats}")
        return True

    def run_matching(self, database_path: Path) -> bool:
        """Runs COLMAP feature matching."""
        logger.info(f"Running {self.match_settings.strategy} matching...")
        
        if self.match_settings.strategy == "exhaustive":
            success = self.runner.exhaustive_matcher(database_path=database_path)
        else:
            logger.error(f"Matching strategy {self.match_settings.strategy} not supported yet.")
            return False
            
        if not success:
            logger.error("Feature matching failed.")
            return False
            
        stats = get_database_statistics(database_path)
        logger.info(f"Database after matching: {stats}")
        return True

    def run_mapper(self, image_dir: Path, database_path: Path, output_dir: Path, mapper_options: dict = None) -> dict:
        """Runs COLMAP mapper and returns post-mapper statistics for all models."""
        sparse_dir = output_dir / "sparse"
        if sparse_dir.exists():
            shutil.rmtree(sparse_dir)
        sparse_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Running sparse reconstruction (mapper)...")
        success = self.runner.mapper(
            database_path=database_path,
            image_path=image_dir,
            output_path=sparse_dir,
            mapper_options=mapper_options
        )
        if not success:
            logger.error("Sparse reconstruction failed.")
            return {"success": False}
            
        models_stats = []
        for model_dir in sparse_dir.iterdir():
            if not model_dir.is_dir() or not model_dir.name.isdigit():
                continue
                
            model_id = int(model_dir.name)
            if not (model_dir / "points3D.bin").exists() and not (model_dir / "points3D.txt").exists():
                continue
                
            txt_dir = output_dir / f"sparse_txt_{model_id}"
            txt_dir.mkdir(exist_ok=True)
            self.runner.model_converter(input_path=model_dir, output_path=txt_dir, output_type="TXT")
            
            reg_img = 0
            pts = 0
            images_txt = txt_dir / "images.txt"
            points_txt = txt_dir / "points3D.txt"
            
            if images_txt.exists():
                with open(images_txt, "r") as f:
                    reg_img = sum(1 for line in f if not line.startswith("#") and line.strip() != "") // 2
            if points_txt.exists():
                min_x = min_y = min_z = float('inf')
                max_x = max_y = max_z = float('-inf')
                with open(points_txt, "r") as f:
                    for line in f:
                        if not line.startswith("#"):
                            pts += 1
                            parts = line.strip().split()
                            if len(parts) >= 4:
                                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                                min_x = min(min_x, x)
                                min_y = min(min_y, y)
                                min_z = min(min_z, z)
                                max_x = max(max_x, x)
                                max_y = max(max_y, y)
                                max_z = max(max_z, z)
                if pts > 0:
                    extents = [max_x - min_x, max_y - min_y, max_z - min_z]
                    largest_extent = max(extents)
                    smallest_extent = min(extents)
                    extent_ratio = largest_extent / smallest_extent if smallest_extent > 0 else float('inf')
                    bbox = {
                        "extents": extents,
                        "largest_extent": largest_extent,
                        "smallest_extent": smallest_extent,
                        "ratio": extent_ratio,
                        "degenerate": extent_ratio > 20.0  # simple heuristic for thin line/plane
                    }
                else:
                    bbox = None
            else:
                bbox = None
                    
            shutil.rmtree(txt_dir, ignore_errors=True)
            
            models_stats.append({
                "model_id": model_id,
                "registered_images": reg_img,
                "sparse_points": pts,
                "bbox": bbox,
                "path": str(model_dir)
            })
            
        models_stats.sort(key=lambda x: x["registered_images"], reverse=True)
        largest_model = models_stats[0] if models_stats else None
        
        if largest_model:
            ply_path = output_dir / "sparse_point_cloud.ply"
            logger.info(f"Converting largest model ({largest_model['model_id']}) to PLY format...")
            self.runner.model_converter(
                input_path=Path(largest_model['path']),
                output_path=ply_path,
                output_type="PLY"
            )
            
        return {
            "success": True,
            "models": models_stats,
            "largest_model": largest_model,
            "registered_images": largest_model["registered_images"] if largest_model else 0,
            "sparse_points": largest_model["sparse_points"] if largest_model else 0,
            "database_stats": get_database_statistics(database_path)
        }
