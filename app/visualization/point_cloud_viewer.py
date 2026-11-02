import open3d as o3d
from pathlib import Path
from loguru import logger

class PointCloudViewer:
    @staticmethod
    def show(ply_path: str | Path):
        """Visualizes a PLY file using Open3D."""
        ply_path = Path(ply_path)
        if not ply_path.exists():
            logger.error(f"Point cloud file not found: {ply_path}")
            return
            
        logger.info(f"Loading point cloud from {ply_path}")
        pcd = o3d.io.read_point_cloud(str(ply_path))
        
        if pcd.is_empty():
            logger.error("Point cloud is empty.")
            return
            
        logger.info("Opening 3D viewer...")
        o3d.visualization.draw_geometries(
            [pcd],
            window_name="AI Dental 3D Reconstruction Viewer",
            width=1024,
            height=768,
            left=50,
            top=50,
            point_show_normal=False
        )
