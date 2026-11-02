import subprocess
from pathlib import Path
import time
from loguru import logger
import json
import shutil

class ColmapRunner:
    def __init__(self, colmap_path: str = "colmap"):
        p = Path(colmap_path)
        if p.is_absolute() and p.exists():
            resolved = str(p)
        else:
            resolved = shutil.which(colmap_path)
            
        if not resolved:
            raise FileNotFoundError(f"COLMAP executable not found. Could not resolve '{colmap_path}' via PATH or as an absolute path.")
            
        self.colmap_path = resolved
        logger.info(f"COLMAP executable: {self.colmap_path}")

    def _run_command(self, cmd: list[str]) -> tuple[bool, str, str, float]:
        """Runs a subprocess command, returning success, stdout, stderr, and execution time."""
        logger.debug(f"Running command: {' '.join(cmd)}")
        start_time = time.time()
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            exec_time = time.time() - start_time
            return True, result.stdout, result.stderr, exec_time
        except subprocess.CalledProcessError as e:
            exec_time = time.time() - start_time
            logger.error(f"Command failed with exit code {e.returncode}")
            logger.error(f"Stderr: {e.stderr}")
            return False, e.stdout, e.stderr, exec_time
        except FileNotFoundError:
            exec_time = time.time() - start_time
            logger.error(f"Executable not found: {cmd[0]}")
            return False, "", "Executable not found", exec_time

    def feature_extractor(self, database_path: Path, image_path: Path, camera_model: str = "SIMPLE_RADIAL", single_camera: bool = True) -> bool:
        cmd = [
            self.colmap_path, "feature_extractor",
            "--database_path", str(database_path),
            "--image_path", str(image_path),
            "--ImageReader.camera_model", camera_model,
            "--ImageReader.single_camera", "1" if single_camera else "0"
        ]
        success, _, _, _ = self._run_command(cmd)
        return success

    def exhaustive_matcher(self, database_path: Path) -> bool:
        cmd = [
            self.colmap_path, "exhaustive_matcher",
            "--database_path", str(database_path)
        ]
        success, _, _, _ = self._run_command(cmd)
        return success

    def mapper(self, database_path: Path, image_path: Path, output_path: Path, mapper_options: dict = None) -> bool:
        output_path.mkdir(parents=True, exist_ok=True)
        cmd = [
            self.colmap_path, "mapper",
            "--database_path", str(database_path),
            "--image_path", str(image_path),
            "--output_path", str(output_path)
        ]
        
        if mapper_options:
            for key, value in mapper_options.items():
                if value is not None:
                    # Map python snake_case to COLMAP's expected argument case (e.g. init_min_num_inliers)
                    cmd.extend([f"--Mapper.{key}", str(value)])
                    
        success, _, _, _ = self._run_command(cmd)
        return success

    def model_converter(self, input_path: Path, output_path: Path, output_type: str = "PLY") -> bool:
        """Converts COLMAP bin/txt model to PLY or other formats."""
        cmd = [
            self.colmap_path, "model_converter",
            "--input_path", str(input_path),
            "--output_path", str(output_path),
            "--output_type", output_type
        ]
        success, _, _, _ = self._run_command(cmd)
        return success
