import json
from pathlib import Path
from datetime import datetime

class QualityReportGenerator:
    def __init__(self, settings):
        self.settings = settings
        
    def generate_report(self, 
                        output_dir: Path,
                        preprocessing_results: list[dict],
                        feature_stats: dict,
                        match_stats: dict,
                        mapper_stats: dict = None):
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        total_input_images = len(preprocessing_results)
        
        # 2D stats
        blurry_count = sum(1 for r in preprocessing_results if r.get("blur_class") == "BLURRY")
        rejected_count = sum(1 for r in preprocessing_results if r.get("status") == "rejected")
        warning_count = sum(1 for r in preprocessing_results if r.get("status") == "warning")
        accepted_count = sum(1 for r in preprocessing_results if r.get("status") == "accepted")
        
        blur_scores = [r["blur_score"] for r in preprocessing_results if "blur_score" in r]
        avg_blur = sum(blur_scores)/len(blur_scores) if blur_scores else 0
        min_blur = min(blur_scores) if blur_scores else 0
        max_blur = max(blur_scores) if blur_scores else 0
        
        # Feature stats
        feat_counts = list(feature_stats.values())
        avg_feat = sum(feat_counts)/len(feat_counts) if feat_counts else 0
        min_feat = min(feat_counts) if feat_counts else 0
        max_feat = max(feat_counts) if feat_counts else 0
        low_feat_count = sum(1 for c in feat_counts if c < self.settings.quality.min_keypoints)
        
        # Match stats
        pairs = match_stats.get("pairs", [])
        num_pairs = len(pairs)
        weak_pairs = sum(1 for p in pairs if not p["is_strong"])
        total_verified_matches = sum(p["match_count"] for p in pairs)
        total_raw_matches = sum(p.get("raw_matches", 0) for p in pairs)
        largest_comp_size = match_stats.get("largest_component_size", 0)
        disconnected_count = len(match_stats.get("disconnected_images", []))
        
        # Mapper stats (if run)
        reg_images = mapper_stats.get("registered_images", 0) if mapper_stats else 0
        sparse_points = mapper_stats.get("sparse_points", 0) if mapper_stats else 0
        reg_percentage = (reg_images / total_input_images * 100) if total_input_images > 0 else 0
        
        # Determine Recommendation
        recommendation = "READY_FOR_DENSE_RECONSTRUCTION"
        evidence = []
        
        temporal = match_stats.get("temporal_analysis", {})
        gap1_avg = temporal.get(1, {}).get("avg", 0)
        
        if total_input_images < self.settings.quality.min_images:
            recommendation = "INSUFFICIENT_IMAGE_OVERLAP"
            evidence.append(f"Only {total_input_images} input images provided (minimum {self.settings.quality.min_images} required).")
            
        elif rejected_count > (total_input_images * 0.5):
            recommendation = "REVIEW_IMAGE_QUALITY"
            evidence.append(f"{rejected_count} out of {total_input_images} images were rejected during 2D analysis.")
            
        elif avg_feat >= 500 and gap1_avg >= 100 and reg_images < (0.8 * total_input_images) and mapper_stats:
            recommendation = "REGISTRATION_LIMITED_RECONSTRUCTION"
            evidence.append(f"Feature counts (avg {avg_feat:.0f}) and temporal matching (avg {gap1_avg:.0f} for gap 1) are adequate.")
            evidence.append(f"However, only {reg_images} out of {total_input_images} images successfully registered.")
            
        elif low_feat_count > (total_input_images * 0.5):
            recommendation = "INSUFFICIENT_FEATURES"
            evidence.append(f"{low_feat_count} images have fewer than {self.settings.quality.min_keypoints} features.")
            
        elif disconnected_count > (total_input_images * 0.2):
            recommendation = "INSUFFICIENT_IMAGE_OVERLAP"
            evidence.append(f"{disconnected_count} images are disconnected from the main graph.")
            evidence.append(f"Largest connected component has only {largest_comp_size} images.")
            
        elif weak_pairs > (num_pairs * 0.5) and num_pairs > 0:
            recommendation = "INSUFFICIENT_MATCHING"
            evidence.append(f"Over 50% of matching pairs are weak (< {self.settings.quality.min_matches} matches or < {self.settings.quality.min_match_ratio*100}% ratio).")
            
        elif mapper_stats:
            if reg_images < 10:
                recommendation = "INSUFFICIENT_CAMERA_REGISTRATION"
                evidence.append(f"Only {reg_images} images registered (minimum 10 required for stable geometry).")
            elif sparse_points < 1000:
                recommendation = "SPARSE_GEOMETRY_WEAK"
                evidence.append(f"Only {sparse_points} sparse points generated (minimum 1000 required).")
            else:
                recommendation = "SPARSE_RECONSTRUCTION_VALIDATION_REQUIRED"
                evidence.append("Sparse reconstruction succeeded, but requires validation before dense processing.")
                
        if not evidence and recommendation == "READY_FOR_DENSE_RECONSTRUCTION":
            evidence.append("All quality thresholds met.")
            if mapper_stats:
                evidence.append(f"{reg_images} images successfully registered with {sparse_points} sparse points.")

        # Build JSON Report
        report_data = {
            "timestamp": datetime.now().isoformat(),
            "recommendation": recommendation,
            "evidence": evidence,
            "statistics": {
                "input_images": total_input_images,
                "accepted_images": accepted_count,
                "warning_images": warning_count,
                "rejected_images": rejected_count,
                "blur": {
                    "blurry_images": blurry_count,
                    "average_score": avg_blur,
                    "min_score": min_blur,
                    "max_score": max_blur
                },
                "features": {
                    "average_count": avg_feat,
                    "min_count": min_feat,
                    "max_count": max_feat,
                    "low_feature_images": low_feat_count
                },
                "matching": {
                    "image_pairs_processed": num_pairs,
                    "total_raw_matches": total_raw_matches,
                    "total_verified_matches": total_verified_matches,
                    "weak_pairs": weak_pairs,
                    "largest_connected_component": largest_comp_size,
                    "disconnected_images": disconnected_count
                },
                "reconstruction": {
                    "registered_images": reg_images,
                    "registration_percentage": reg_percentage,
                    "sparse_points": sparse_points
                } if mapper_stats else None
            },
            "image_details": preprocessing_results
        }
        
        json_path = output_dir / "quality_report.json"
        with open(json_path, "w") as f:
            json.dump(report_data, f, indent=2)
            
        # Build TXT Report
        txt_path = output_dir / "quality_report.txt"
        with open(txt_path, "w") as f:
            f.write("========================================\n")
            f.write("RECONSTRUCTION VALIDATION & DIAGNOSTIC REPORT V2\n")
            f.write("========================================\n\n")
            
            f.write("--- IMAGE QUALITY ---\n")
            f.write(f"Total images:          {total_input_images}\n")
            f.write(f"Accepted:              {accepted_count}\n")
            f.write(f"Warnings:              {warning_count}\n")
            f.write(f"Rejected:              {rejected_count}\n")
            f.write(f"Average blur score:    {avg_blur:.2f}\n")
            f.write(f"Min / Max blur score:  {min_blur:.2f} / {max_blur:.2f}\n")
            f.write(f"Average keypoints:     {avg_feat:.0f}\n")
            f.write(f"Min / Max keypoints:   {min_feat} / {max_feat}\n\n")
            
            f.write("--- MATCHING QUALITY ---\n")
            f.write(f"Image pairs processed: {num_pairs}\n")
            f.write(f"Total raw matches:     {total_raw_matches}\n")
            f.write(f"Total verified matches:{total_verified_matches}\n")
            
            v_matches = [p["match_count"] for p in pairs] if pairs else []
            avg_vm = sum(v_matches) / len(v_matches) if v_matches else 0
            med_vm = sorted(v_matches)[len(v_matches)//2] if v_matches else 0
            
            f.write(f"Avg verified matches:  {avg_vm:.1f}\n")
            f.write(f"Median verified matches:{med_vm}\n")
            f.write(f"Strong pair count:     {num_pairs - weak_pairs}\n")
            f.write(f"Weak pair count:       {weak_pairs}\n")
            f.write(f"Disconnected images:   {disconnected_count}\n\n")
            
            temporal = match_stats.get("temporal_analysis")
            if temporal:
                f.write("--- TEMPORAL MATCHING ---\n")
                f.write("Gap | Pairs | Avg | Median | Min | Max | Zero-match %\n")
                for gap in [1, 2, 3, 5, 10]:
                    t = temporal.get(gap)
                    if t and t["pairs"] > 0:
                        f.write(f"{gap:>3} | {t['pairs']:>5} | {t['avg']:>3.0f} | {t['median']:>6.0f} | {t['min']:>3} | {t['max']:>3} | {t['zero_pct']:>11.1f}%\n")
                f.write("\n")
            
            if mapper_stats:
                f.write("--- RECONSTRUCTION QUALITY ---\n")
                models = mapper_stats.get("models", [])
                if not models:
                    f.write("No models were successfully reconstructed.\n")
                for m in models:
                    f.write(f"Model ID:              {m['model_id']}\n")
                    f.write(f"Registered images:     {m['registered_images']}\n")
                    f.write(f"Registration %:        {(m['registered_images'] / total_input_images * 100):.1f}%\n")
                    f.write(f"Sparse points:         {m['sparse_points']}\n")
                    pts_per_reg = m['sparse_points'] / m['registered_images'] if m['registered_images'] > 0 else 0
                    f.write(f"Points/reg image:      {pts_per_reg:.1f}\n")
                    bbox = m.get("bbox")
                    if bbox:
                        f.write(f"Bounding box extents:  [{bbox['extents'][0]:.2f}, {bbox['extents'][1]:.2f}, {bbox['extents'][2]:.2f}]\n")
                        f.write(f"Bounding box ratio:    {bbox['ratio']:>5.2f}\n")
                        f.write(f"Degenerate flag:       {bbox['degenerate']}\n")
                    f.write("\n")
                
            f.write("========================================\n")
            f.write(f"RESULT: {recommendation}\n\n")
            f.write("Evidence:\n")
            for ev in evidence:
                f.write(f"- {ev}\n")
            f.write("========================================\n\n")
            
            f.write("--- NEXT ACTION ---\n")
            if recommendation == "REGISTRATION_LIMITED_RECONSTRUCTION":
                f.write(f"Feature extraction and sequential matching are not the primary bottleneck. Consecutive images have strong verified geometric correspondence, but camera registration stops at {reg_images}/{total_input_images} images. Camera-model experiments also produce the same registration count. The next investigation should focus on camera calibration / intrinsic accuracy and reconstruction geometry rather than increasing feature count.\n\n")
            elif recommendation in ["INSUFFICIENT_CAMERA_REGISTRATION", "SPARSE_GEOMETRY_WEAK", "RECONSTRUCTION_FAILED"]:
                f.write("Dense reconstruction skipped: sparse reconstruction is insufficient for reliable dense reconstruction.\n\n")
            else:
                f.write("Proceed with evaluation or dense reconstruction.\n\n")
            
            f.write("--- Dental-Specific Capture Guidance ---\n")
            f.write("Dental surfaces can be difficult for traditional photogrammetry because they may contain smooth surfaces, similar colors, specular reflections, and limited texture.\n")
            f.write("Recommendations:\n")
            f.write("- Use consistent, diffuse lighting to avoid harsh specular reflections.\n")
            f.write("- Maintain a consistent camera distance.\n")
            f.write("- Ensure sufficient overlap (70-80%) between consecutive images.\n")
            f.write("- Move the camera approximately 10-15 degrees between frames.\n")
            f.write("- Aim for 30-50 images for a complete capture when practical.\n")
            
        return report_data

    def generate_experiment_report(self, output_dir: Path, total_input_images: int, match_stats: dict, experiment_results: dict):
        report_data = {
            "timestamp": datetime.now().isoformat(),
            "experiments": experiment_results
        }
        
        json_path = output_dir / "experiment_report.json"
        with open(json_path, "w") as f:
            json.dump(report_data, f, indent=2)
            
        txt_path = output_dir / "experiment_report.txt"
        with open(txt_path, "w") as f:
            f.write("==================================================\n")
            f.write("CAMERA MODEL EXPERIMENT\n")
            f.write("==================================================\n\n")
            f.write(f"Input images: {total_input_images}\n\n")
            f.write("Features/matches reused:\nYES\n\n")
            f.write("Feature extraction repeated:\nNO\n\n")
            f.write("Matching repeated:\nNO\n\n")
            f.write("IMPORTANT:\n")
            f.write("Camera intrinsics for RADIAL and OPENCV are approximated from the\n")
            f.write("original SIMPLE_RADIAL camera and are not independently calibrated.\n")
            f.write("Results are therefore comparative experiments, not calibrated camera-model evaluation.\n\n")
            
            # Ranking based on reg_percentage, sparse_points, and model fragmentation
            def rank_score(item):
                stats = item[1]
                if not stats.get("success"):
                    return (0, 0, float('-inf'))
                reg = stats.get("registered_images", 0)
                pts = stats.get("sparse_points", 0)
                models_count = len(stats.get("models", []))
                # higher reg is better, higher pts is better, lower models_count is better
                return (reg, pts, -models_count)
                
            ranked_results = sorted(experiment_results.items(), key=rank_score, reverse=True)
            best_model_name = ranked_results[0][0] if ranked_results and ranked_results[0][1].get("success") else None
            best_stats = ranked_results[0][1] if best_model_name else None
            
            f.write("--- COMPARISON TABLE ---\n")
            f.write("Model           | Registered | Registration % | Sparse Points | Points/Reg Image | Models\n")
            f.write("-" * 95 + "\n")
            for model_name, stats in ranked_results:
                if not stats.get("success"):
                    f.write(f"{model_name:<15} | {'Failed':>10} | {'-':>14} | {'-':>13} | {'-':>16} | {'-':>6}\n")
                    continue
                reg = stats.get("registered_images", 0)
                reg_pct = (reg / total_input_images) * 100
                pts = stats.get("sparse_points", 0)
                pts_per_reg = pts / reg if reg > 0 else 0
                models_count = len(stats.get("models", []))
                f.write(f"{model_name:<15} | {reg:>10} | {reg_pct:>13.1f}% | {pts:>13} | {pts_per_reg:>16.1f} | {models_count:>6}\n")
            f.write("\n")
            
            for model_name, stats in ranked_results:
                f.write("-" * 50 + "\n")
                f.write(f"{model_name}\n")
                f.write("-" * 50 + "\n")
                if not stats.get("success"):
                    f.write("Mapper failed.\n\n")
                    continue
                    
                reg_images = stats.get("registered_images", 0)
                sparse_points = stats.get("sparse_points", 0)
                f.write(f"Registered images: {reg_images}/{total_input_images}\n")
                f.write(f"Sparse points: {sparse_points}\n")
                
                models = stats.get("models", [])
                f.write(f"Models: {len(models)}\n")
                
                largest = stats.get("largest_model")
                if largest:
                    f.write(f"Largest model: ID {largest['model_id']} ({largest['registered_images']} imgs, {largest['sparse_points']} pts)\n")
                f.write("\n")
                    
            f.write("-" * 50 + "\n")
            f.write("BEST EXPERIMENTAL RESULT\n")
            f.write("-" * 50 + "\n")
            
            if best_model_name:
                f.write(f"{best_model_name} produced the strongest result in this controlled experiment with {best_stats.get('registered_images', 0)} registered images and {best_stats.get('sparse_points', 0)} sparse points.\n\n")
                f.write(f"IMPORTANT: {best_model_name} was evaluated using approximated intrinsics derived from SIMPLE_RADIAL. This is comparative evidence only and is not a calibrated camera-model conclusion.\n\n")
            else:
                f.write("No successful models produced.\n\n")
                
            f.write("NEXT ACTION:\n")
            temporal = match_stats.get("temporal_analysis")
            gap1_avg = temporal.get(1, {}).get("avg", 0) if temporal else 0
            
            if gap1_avg < 20:
                f.write("Capture sequence likely has insufficient frame-to-frame overlap or insufficient stable visual texture.\n")
            else:
                f.write("Feature extraction and sequential matching are not the primary bottleneck. Consecutive images have strong verified geometric correspondence, but camera registration stops early. Camera-model experiments also produce similar registration counts. The next investigation should focus on camera calibration / intrinsic accuracy and reconstruction geometry rather than increasing feature count.\n")
