# AI-Based 3D Dental Reconstruction

## Project Purpose
This project is a research/prototype system that takes multiple overlapping 2D intraoral photographs of teeth and reconstructs them into a 3D dental model using Structure-from-Motion (SfM) and Dense Multi-View Stereo.

**Note:** This is a prototype system. Accurate clinical dental measurements are NOT currently guaranteed.

## Current Milestone (Milestone 1)
The current milestone establishes the foundation of the pipeline and implements the initial sparse 3D reconstruction using COLMAP.

Pipeline steps included:
1. Image validation
2. Image quality assessment
3. Image preprocessing
4. COLMAP Structure-from-Motion
5. Sparse point cloud visualization

## Architecture
The project is built in Python with a modular architecture to allow future deep learning integrations (e.g., YOLO for segmentation, SuperPoint for feature extraction).

## Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### COLMAP Installation
COLMAP is an external dependency required for the reconstruction phase.
1. Download COLMAP from [https://colmap.github.io/](https://colmap.github.io/)
2. Add the COLMAP executable to your system `PATH`.
3. Verify installation using `python scripts/check_environment.py`.

## Dataset Structure
Place your datasets in the `data/raw/` directory, grouped by case:
```
data/raw/case_001/
    ├── 001.jpg
    ├── 002.jpg
    └── 003.jpg
```

## Running the Pipeline

To run the full reconstruction pipeline on a specific case:
```bash
python run_pipeline.py --input data/raw/case_001
```

Options:
- `--visualize`: Open the 3D viewer after successful reconstruction.
- `--skip-preprocessing`: Skip image preprocessing steps.
- `--skip-quality`: Skip image quality assessment.
- `--config <path>`: Specify a custom configuration file.
- `--verbose`: Enable detailed logging.

## Viewing Reconstruction
To view an existing point cloud without re-running the pipeline:
```bash
python run_pipeline.py --view data/reconstruction/case_001/sparse_point_cloud.ply
```

## Output Files
The pipeline generates the following outputs:
- **`data/processed/<case>`**: Preprocessed images and `quality_report.json`.
- **`data/reconstruction/<case>`**: COLMAP database, binary outputs, `sparse_point_cloud.ply`, and `reconstruction_report.json`.
- **`logs/`**: Pipeline execution logs.

## Troubleshooting
- **COLMAP not found**: Ensure COLMAP is added to your `PATH` environment variable.
- **Sparse reconstruction failed**: Check the logs and `reconstruction_report.json` for details. Ensure you have sufficient overlapping images.
- **Low valid image count**: Review the `quality_report.json` to see why images were rejected or flagged with warnings.

## Future Milestones
- **Milestone 2**: Image Quality + Tooth Segmentation (YOLO)
- **Milestone 3**: Learned Feature Extraction (SuperPoint / DINOv2)
- **Milestone 4**: Learned Feature Matching (LightGlue)
- **Milestone 5**: Dense Reconstruction (COLMAP MVS / learned MVS)
- **Milestone 6**: Point Cloud Registration
- **Milestone 7**: Point Cloud Fusion
- **Milestone 8**: Mesh Reconstruction (Poisson)
- **Milestone 9**: Mesh Refinement (GNN)
- **Milestone 10**: Interactive Application
