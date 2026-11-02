import sqlite3
from pathlib import Path
from loguru import logger
import collections
import sys

class ColmapAnalyzer:
    def __init__(self, db_path: Path):
        self.db_path = db_path

    def _pair_id_to_image_ids(self, pair_id: int):
        image_id2 = pair_id % 2147483647
        image_id1 = (pair_id - image_id2) // 2147483647
        return image_id1, image_id2

    def _get_matrix_rows(self, rows, cols, data, byte_size=4):
        """Calculates matrix rows, falling back to BLOB size if necessary."""
        if rows is not None and rows > 0:
            return rows
        if data is not None and cols is not None and cols > 0:
            return len(data) // (cols * byte_size)
        return 0

    def analyze_features(self) -> dict:
        """Retrieves actual feature counts by inspecting matrix dimensions."""
        if not self.db_path.exists():
            return {}

        feature_stats = {}
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT image_id, name FROM images")
                image_map = {row[0]: row[1] for row in cursor.fetchall()}
                
                try:
                    cursor.execute("SELECT image_id, rows, cols, data FROM keypoints")
                    for image_id, rows, cols, data in cursor.fetchall():
                        if image_id in image_map:
                            img_name = image_map[image_id]
                            # Keypoints are typically float32 (4 bytes)
                            num_features = self._get_matrix_rows(rows, cols, data, byte_size=4)
                            feature_stats[img_name] = num_features
                except sqlite3.OperationalError:
                    logger.warning("Keypoints table does not exist or has unexpected schema.")
        except Exception as e:
            logger.error(f"Failed to analyze features: {e}")
            
        return feature_stats

    def analyze_matches(self, feature_stats: dict, min_matches: int, min_match_ratio: float) -> dict:
        """Analyzes both raw and verified matches and graph connectivity."""
        if not self.db_path.exists():
            return {"pairs": [], "connected_components": [], "largest_component_size": 0, "disconnected_images": []}

        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT image_id, name FROM images")
                image_map = {row[0]: row[1] for row in cursor.fetchall()}
        except Exception:
            return {"pairs": [], "connected_components": [], "largest_component_size": 0, "disconnected_images": []}

        # Get raw matches
        raw_matches_map = {}
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT pair_id, rows, cols, data FROM matches")
                for pair_id, rows, cols, data in cursor.fetchall():
                    num_raw = self._get_matrix_rows(rows, cols, data, byte_size=4)
                    raw_matches_map[pair_id] = num_raw
        except sqlite3.OperationalError:
            pass

        pairs = []
        adj_list = collections.defaultdict(list)
        all_nodes = set(image_map.values())
        
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT pair_id, rows, cols, data FROM two_view_geometries")
                
                for pair_id, rows, cols, data in cursor.fetchall():
                    id1, id2 = self._pair_id_to_image_ids(pair_id)
                    if id1 not in image_map or id2 not in image_map:
                        continue
                        
                    img1, img2 = image_map[id1], image_map[id2]
                    
                    feat1 = feature_stats.get(img1, 0)
                    feat2 = feature_stats.get(img2, 0)
                    min_feat = min(feat1, feat2)
                    
                    num_verified = self._get_matrix_rows(rows, cols, data, byte_size=4)
                    num_raw = raw_matches_map.get(pair_id, 0)
                    
                    if num_verified == 0:
                        continue
                        
                    match_ratio = num_verified / min_feat if min_feat > 0 else 0.0
                    is_strong = (num_verified >= min_matches) and (match_ratio >= min_match_ratio)
                    
                    pairs.append({
                        "image_A": img1,
                        "image_B": img2,
                        "match_count": num_verified,
                        "raw_matches": num_raw,
                        "match_ratio": match_ratio,
                        "is_strong": is_strong
                    })
                    
                    if is_strong:
                        adj_list[img1].append(img2)
                        adj_list[img2].append(img1)
                        
        except sqlite3.OperationalError:
            logger.warning("two_view_geometries table does not exist yet.")
        except Exception as e:
            logger.error(f"Failed to analyze matches: {e}")

        # Compute sequential and temporal gap statistics
        sorted_images = sorted(list(all_nodes))
        img_to_idx = {img: idx for idx, img in enumerate(sorted_images)}
        
        # We need to consider all possible pairs for these gaps to count zero-match pairs
        # A pair exists if both images are in all_nodes.
        gaps_to_track = [1, 2, 3, 5, 10]
        gap_stats = {g: {"matches": [], "total_possible": 0} for g in gaps_to_track}
        
        for g in gaps_to_track:
            for i in range(len(sorted_images) - g):
                gap_stats[g]["total_possible"] += 1
                
        for p in pairs:
            idx1 = img_to_idx.get(p["image_A"])
            idx2 = img_to_idx.get(p["image_B"])
            if idx1 is not None and idx2 is not None:
                gap = abs(idx1 - idx2)
                if gap in gap_stats:
                    gap_stats[gap]["matches"].append(p["match_count"])
                    
        temporal_analysis = {}
        for g in gaps_to_track:
            m = gap_stats[g]["matches"]
            total = gap_stats[g]["total_possible"]
            zeros = total - len(m)
            # Add zeros to the list for accurate median/avg computation
            m_full = m + [0] * zeros
            if m_full:
                m_full.sort()
                avg_m = sum(m_full) / len(m_full)
                med_m = m_full[len(m_full)//2]
                min_m = m_full[0]
                max_m = m_full[-1]
            else:
                avg_m = med_m = min_m = max_m = 0
                
            zero_pct = (zeros / total * 100) if total > 0 else 0
            
            # For gap=1, we also need bins
            bins = {"<20": 0, "20-50": 0, "50-100": 0, "100-200": 0, ">200": 0}
            if g == 1:
                for match_cnt in m_full:
                    if match_cnt < 20: bins["<20"] += 1
                    elif match_cnt < 50: bins["20-50"] += 1
                    elif match_cnt < 100: bins["50-100"] += 1
                    elif match_cnt < 200: bins["100-200"] += 1
                    else: bins[">200"] += 1
            
            temporal_analysis[g] = {
                "pairs": total,
                "avg": avg_m,
                "median": med_m,
                "min": min_m,
                "max": max_m,
                "zeros": zeros,
                "zero_pct": zero_pct,
                "bins": bins if g == 1 else None
            }

        # Compute connected components (DFS)
        visited = set()
        components = []
        
        for node in all_nodes:
            if node not in visited:
                comp = []
                stack = [node]
                while stack:
                    curr = stack.pop()
                    if curr not in visited:
                        visited.add(curr)
                        comp.append(curr)
                        stack.extend(n for n in adj_list[curr] if n not in visited)
                components.append(comp)

        components.sort(key=len, reverse=True)
        largest_size = len(components[0]) if components else 0
        disconnected = []
        if components and len(components) > 1:
            for comp in components[1:]:
                disconnected.extend(comp)

        return {
            "pairs": pairs,
            "connected_components": components,
            "largest_component_size": largest_size,
            "disconnected_images": disconnected,
            "temporal_analysis": temporal_analysis
        }

    def print_diagnostics(self, feature_stats: dict, match_stats: dict):
        """Prints explicit database sanity checks to the console."""
        num_images = len(feature_stats)
        pairs = match_stats.get("pairs", [])
        num_pairs = len(pairs)
        
        feat_counts = list(feature_stats.values())
        avg_feat = sum(feat_counts) / len(feat_counts) if feat_counts else 0
        min_feat = min(feat_counts) if feat_counts else 0
        max_feat = max(feat_counts) if feat_counts else 0
        
        match_counts = [p["match_count"] for p in pairs]
        avg_matches = sum(match_counts) / len(match_counts) if match_counts else 0
        
        # median verified matches/pair and strong-pair matches
        if match_counts:
            sorted_m = sorted(match_counts)
            median_matches = sorted_m[len(sorted_m)//2]
        else:
            median_matches = 0
            
        strong_matches = [p["match_count"] for p in pairs if p["is_strong"]]
        if strong_matches:
            sorted_sm = sorted(strong_matches)
            median_strong = sorted_sm[len(sorted_sm)//2]
        else:
            median_strong = 0
            
        strong_pairs = len(strong_matches)
        weak_pairs = num_pairs - strong_pairs
        disconnected = len(match_stats.get("disconnected_images", []))
        
        print("\n" + "=" * 50)
        print("COLMAP DATABASE DIAGNOSTICS")
        print("=" * 50)
        print(f"Images: {num_images}")
        print(f"Image-pair records: {num_pairs}\n")
        print(f"Average keypoints/image: {avg_feat:.0f}")
        print(f"Minimum keypoints/image: {min_feat}")
        print(f"Maximum keypoints/image: {max_feat}\n")
        print(f"Average verified matches/pair: {avg_matches:.0f}")
        print(f"Median verified matches/pair: {median_matches:.0f}")
        print(f"Median strong-pair verified matches: {median_strong:.0f}\n")
        print(f"Strongly connected pairs: {strong_pairs}")
        print(f"Weak pairs: {weak_pairs}")
        print(f"Disconnected images: {disconnected}\n")
        
        temporal = match_stats.get("temporal_analysis")
        if temporal:
            print("--- TEMPORAL GAP MATCHING (Verified Geometric Matches) ---")
            print("Gap | Pairs | Avg | Median | Min | Max | Zero-match %")
            for gap in [1, 2, 3, 5, 10]:
                t = temporal.get(gap)
                if t and t["pairs"] > 0:
                    print(f"{gap:>3} | {t['pairs']:>5} | {t['avg']:>3.0f} | {t['median']:>6.0f} | {t['min']:>3} | {t['max']:>3} | {t['zero_pct']:>11.1f}%")
            print("=" * 50 + "\n")
