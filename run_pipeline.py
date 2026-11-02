import argparse
import sys
from pathlib import Path
import shutil

from app.config.settings import Settings
from app.utils.logger import setup_logger
from app.quality.image_quality import ImageQualityAnalyzer
from app.preprocessing.image_processor import ImageProcessor
from app.reconstruction.reconstruction_pipeline import ReconstructionPipeline
from app.reconstruction.colmap_analyzer import ColmapAnalyzer
from app.quality.quality_report import QualityReportGenerator
from app.visualization.point_cloud_viewer import PointCloudViewer
from loguru import logger

def main():
    parser = argparse.ArgumentParser(description="AI Dental 3D Reconstruction Pipeline")
    parser.add_argument("--input", type=str, help="Path to raw image case directory (e.g. data/raw/case_001)")
    parser.add_argument("--visualize", action="store_true", help="Visualize the reconstruction when finished")
    parser.add_argument("--view", type=str, help="Path to existing .ply file to view")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--experiment", action="store_true", help="Run camera-model experiments (SIMPLE_RADIAL, RADIAL, OPENCV)")
    parser.add_argument("--mapper-experiment", action="store_true", help="Run mapper-parameter experiments (not implemented yet)")
    
    args = parser.parse_args()
    
    if args.view:
        PointCloudViewer.show(args.view)
        return
        
    if not args.input:
        parser.error("--input is required unless using --view")
        
    try:
        settings = Settings.load(args.config)
    except Exception as e:
        print(f"Failed to load config: {e}")
        sys.exit(1)
        
    log = setup_logger(settings.paths.logs, args.verbose)
    
    input_dir = Path(args.input).resolve()
    if not input_dir.exists() or not input_dir.is_dir():
        log.error(f"Input directory does not exist: {input_dir}")
        sys.exit(1)
        
    case_name = input_dir.name
    processed_dir = (settings.paths.processed_data / case_name).resolve()
    recon_dir = (settings.paths.reconstruction_data / case_name).resolve()
    
    # Ensure processed and recon dirs exist
    if processed_dir.exists() and processed_dir != input_dir:
        shutil.rmtree(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    if recon_dir.exists() and recon_dir != input_dir:
        shutil.rmtree(recon_dir)
    recon_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 52)
    print("AI DENTAL 3D RECONSTRUCTION & DIAGNOSTICS")
    print("=" * 52 + "\n")
    
    # 1. 2D Quality Assessment
    print("[1/6] Running 2D image quality assessment...")
    quality_analyzer = ImageQualityAnalyzer(settings.quality)
    preprocessing_results = quality_analyzer.process_directory(input_dir)
    if not preprocessing_results:
        log.error("Pipeline aborted: No images found or readable.")
        sys.exit(1)
        
    # 2. Preprocessing
    print("[2/6] Preprocessing images (Target MP resizing, etc.)...")
    processor = ImageProcessor(settings.preprocessing)
    success = processor.process_directory(input_dir, processed_dir, preprocessing_results)
    if not success:
        log.error("Pipeline aborted: Preprocessing failed.")
        sys.exit(1)
        
    # Initialize Reconstruction components
    pipeline = ReconstructionPipeline(settings.reconstruction, settings.matching)
    database_path = recon_dir / "database.db"
    if database_path.exists():
        database_path.unlink()
        
    # 3. COLMAP Feature Extraction & Diagnostics
    print("[3/6] Running COLMAP feature extraction...")
    success = pipeline.run_extraction(processed_dir, database_path)
    if not success:
        log.error("Pipeline aborted: Feature extraction failed.")
        sys.exit(1)
        
    colmap_analyzer = ColmapAnalyzer(database_path)
    feature_stats = colmap_analyzer.analyze_features()
    
    # 4. COLMAP Feature Matching & Diagnostics
    print("[4/6] Running COLMAP feature matching...")
    success = pipeline.run_matching(database_path)
    if not success:
        log.error("Pipeline aborted: Feature matching failed.")
        sys.exit(1)
        
    match_stats = colmap_analyzer.analyze_matches(
        feature_stats, 
        settings.quality.min_matches, 
        settings.quality.min_match_ratio
    )
    
    colmap_analyzer.print_diagnostics(feature_stats, match_stats)
    
    if args.experiment:
        print("\n[5/6] Running Experimental Camera-Model Tests...")
        experiments_dir = recon_dir / "experiments"
        experiments_dir.mkdir(parents=True, exist_ok=True)
        
        models_to_test = {
            "SIMPLE_RADIAL": 2,
            "RADIAL": 3,
            "OPENCV": 4
        }
        
        experiment_results = {}
        
        for model_name, model_id in models_to_test.items():
            print(f"\n--- Testing Camera Model: {model_name} ---")
            exp_dir = experiments_dir / model_name.lower()
            exp_dir.mkdir(parents=True, exist_ok=True)
            
            exp_db = exp_dir / "database.db"
            if exp_db.exists():
                exp_db.unlink()
                
            shutil.copy2(database_path, exp_db)
            
            # Modify cameras table using SQLite
            import sqlite3
            import numpy as np
            conn = sqlite3.connect(str(exp_db))
            c = conn.cursor()
            try:
                c.execute("SELECT camera_id, model, width, height, params, prior_focal_length FROM cameras")
                cameras = c.fetchall()
                
                for row in cameras:
                    cam_id, old_model, w, h, params, prior = row
                    param_arr = np.frombuffer(params, dtype=np.float64)
                    
                    if len(param_arr) >= 3:
                        f = param_arr[0]
                        cx = param_arr[1]
                        cy = param_arr[2]
                        k = param_arr[3] if len(param_arr) > 3 else 0.0
                        
                        if model_id == 2:
                            new_params = np.array([f, cx, cy, k], dtype=np.float64)
                        elif model_id == 3:
                            new_params = np.array([f, cx, cy, k, 0.0], dtype=np.float64)
                        elif model_id == 4:
                            new_params = np.array([f, f, cx, cy, k, 0.0, 0.0, 0.0], dtype=np.float64)
                        else:
                            new_params = param_arr
                            
                        c.execute("UPDATE cameras SET model=?, params=? WHERE camera_id=?", (model_id, new_params.tobytes(), cam_id))
                conn.commit()
            except Exception as e:
                log.error(f"Failed to update camera parameters for {model_name}: {e}")
            finally:
                conn.close()
                
            # Run mapper on experiment
            mapper_options = None # Keep defaults for the first experiment
            stats = pipeline.run_mapper(processed_dir, exp_db, exp_dir, mapper_options)
            experiment_results[model_name] = stats
            
        # Generate experiment report
        print("[6/6] Generating Experiment Report...")
        report_gen = QualityReportGenerator(settings)
        report_gen.generate_experiment_report(recon_dir, total_input_images=len(preprocessing_results), match_stats=match_stats, experiment_results=experiment_results)
        
        print("[6/6] Generating final diagnostic report...")
        report = report_gen.generate_report(recon_dir, preprocessing_results, feature_stats, match_stats, mapper_stats=experiment_results.get("SIMPLE_RADIAL"))
        
        mapper_stats = {"success": True} # to satisfy view logic below
        
    else:
        # 5. Sparse Reconstruction (Mapper)
        print("[5/6] Running COLMAP mapper (sparse reconstruction)...")
        mapper_options = settings.mapper.dict() if hasattr(settings, 'mapper') else None
        mapper_stats = pipeline.run_mapper(processed_dir, database_path, recon_dir, mapper_options)
        
        # 6. Quality Report Generation
        print("[6/6] Generating final diagnostic report...")
        report_gen = QualityReportGenerator(settings)
        report = report_gen.generate_report(recon_dir, preprocessing_results, feature_stats, match_stats, mapper_stats)
    
    print("\n" + "=" * 52)
    print("RECONSTRUCTION AND DIAGNOSTICS COMPLETE")
    print("=" * 52)
    print(f"Report: {recon_dir / 'quality_report.txt'}")
    print(f"Output: {recon_dir}")
    print("=" * 52 + "\n")
    
    if args.visualize and mapper_stats.get("success"):
        ply_path = recon_dir / "sparse_point_cloud.ply"
        if ply_path.exists():
            PointCloudViewer.show(ply_path)
        else:
            log.error(f"Expected PLY file not found at {ply_path}")

if __name__ == "__main__":
    main()
