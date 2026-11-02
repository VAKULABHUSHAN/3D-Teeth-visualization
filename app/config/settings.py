from pydantic import BaseModel, Field
import yaml
from pathlib import Path


class ProjectSettings(BaseModel):
    name: str
    version: str


class PathSettings(BaseModel):
    raw_data: Path
    processed_data: Path
    reconstruction_data: Path
    logs: Path


class QualitySettings(BaseModel):
    min_images: int
    min_width: int
    min_height: int
    blur_threshold: float
    brightness_min: float
    brightness_max: float
    auto_reject_blurry: bool = False
    auto_reject_low_feature: bool = False
    min_keypoints: int = 1000
    min_matches: int = 100
    min_match_ratio: float = 0.05
    min_registered_images: int = 8


class PreprocessingResizeSettings(BaseModel):
    enabled: bool


class PreprocessingDenoiseSettings(BaseModel):
    enabled: bool


class PreprocessingClaheSettings(BaseModel):
    enabled: bool
    clip_limit: float
    tile_grid_size: int


class PreprocessingSharpenSettings(BaseModel):
    enabled: bool


class PreprocessingExposureSettings(BaseModel):
    enabled: bool


class PreprocessingSettings(BaseModel):
    enabled: bool
    target_mp: float = 10.0
    resize: PreprocessingResizeSettings = Field(default_factory=lambda: PreprocessingResizeSettings(enabled=True))
    denoise: PreprocessingDenoiseSettings = Field(default_factory=lambda: PreprocessingDenoiseSettings(enabled=False))
    clahe: PreprocessingClaheSettings = Field(default_factory=lambda: PreprocessingClaheSettings(enabled=False, clip_limit=2.0, tile_grid_size=8))
    sharpen: PreprocessingSharpenSettings = Field(default_factory=lambda: PreprocessingSharpenSettings(enabled=False))
    exposure_correction: PreprocessingExposureSettings = Field(default_factory=lambda: PreprocessingExposureSettings(enabled=False))


class ReconstructionSettings(BaseModel):
    colmap_path: str
    camera_model: str
    single_camera: bool


class MatchingSettings(BaseModel):
    strategy: str


class MapperSettings(BaseModel):
    init_min_num_inliers: int = 100
    init_max_error: float = 4.0
    min_num_matches: int = 15
    min_model_size: int = 10
    max_num_models: int = 50


class Settings(BaseModel):
    project: ProjectSettings
    paths: PathSettings
    quality: QualitySettings
    preprocessing: PreprocessingSettings
    reconstruction: ReconstructionSettings
    matching: MatchingSettings
    mapper: MapperSettings = Field(default_factory=MapperSettings)

    @classmethod
    def load(cls, config_path: str | Path = "config.yaml") -> "Settings":
        """Loads configuration from a YAML file."""
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
            
        with open(config_path, "r") as f:
            data = yaml.safe_load(f)
            
        return cls(**data)
