#!/usr/bin/env python3
"""Spectrum similarity scoring methods adapted from RamanLab."""

import numpy as np
from scipy.signal import find_peaks


class SpectrumMatcher:
    """Holds the query spectrum the scoring methods compare against."""

    def __init__(self, wavenumbers, intensities, detected_peaks=None):
        self.current_wavenumbers = np.asarray(wavenumbers, dtype=float)
        self.processed_intensities = np.asarray(intensities, dtype=float)
        self.detected_peaks = detected_peaks

    def calculate_correlation_score(self, db_wavenumbers, db_intensities):
        """Calculate correlation score between current spectrum and database spectrum."""
        try:
            db_wavenumbers = np.array(db_wavenumbers, dtype=float)
            db_intensities = np.array(db_intensities, dtype=float)

            if len(db_wavenumbers) == 0 or len(db_intensities) == 0:
                return 0.0

            common_wavenumbers = np.linspace(
                max(self.current_wavenumbers.min(), db_wavenumbers.min()),
                min(self.current_wavenumbers.max(), db_wavenumbers.max()),
                min(len(self.current_wavenumbers), len(db_wavenumbers))
            )

            query_interp = np.interp(common_wavenumbers, self.current_wavenumbers, self.processed_intensities)
            db_interp = np.interp(common_wavenumbers, db_wavenumbers, db_intensities)

            query_std = np.std(query_interp)
            db_std = np.std(db_interp)

            if query_std == 0 or db_std == 0:
                return 0.0

            query_norm = (query_interp - np.mean(query_interp)) / query_std
            db_norm = (db_interp - np.mean(db_interp)) / db_std

            correlation = np.corrcoef(query_norm, db_norm)[0, 1]

            similarity = abs(correlation)

            return max(0, min(1, similarity))

        except Exception as e:
            print(f"Error in correlation score calculation: {e}")
            return 0.0

    def calculate_multi_window_score(self, db_wavenumbers, db_intensities):
        """Calculate correlation score using multiple diagnostic frequency windows."""
        try:
            db_wavenumbers = np.array(db_wavenumbers, dtype=float)
            db_intensities = np.array(db_intensities, dtype=float)

            if len(db_wavenumbers) == 0 or len(db_intensities) == 0:
                return 0.0

            windows = [
                {"name": "Lattice", "range": (50, 400), "weight": 0.2},
                {"name": "Bending", "range": (400, 800), "weight": 0.3},
                {"name": "Stretching", "range": (800, 1200), "weight": 0.4},
                {"name": "High", "range": (1200, 1800), "weight": 0.1}
            ]

            total_weighted_score = 0.0
            total_weight = 0.0

            for window in windows:
                window_start, window_end = window["range"]

                query_mask = (self.current_wavenumbers >= window_start) & (self.current_wavenumbers <= window_end)
                db_mask = (db_wavenumbers >= window_start) & (db_wavenumbers <= window_end)

                if not np.any(query_mask) or not np.any(db_mask):
                    continue

                overlap_start = max(window_start, 
                                  max(self.current_wavenumbers[query_mask].min(), db_wavenumbers[db_mask].min()))
                overlap_end = min(window_end,
                                min(self.current_wavenumbers[query_mask].max(), db_wavenumbers[db_mask].max()))

                if overlap_end <= overlap_start:
                    continue

                n_points = min(50, np.sum(query_mask), np.sum(db_mask))
                if n_points < 5:
                    continue

                common_wavenumbers = np.linspace(overlap_start, overlap_end, n_points)

                query_interp = np.interp(common_wavenumbers, self.current_wavenumbers, self.processed_intensities)
                db_interp = np.interp(common_wavenumbers, db_wavenumbers, db_intensities)

                query_std = np.std(query_interp)
                db_std = np.std(db_interp)

                if query_std == 0 or db_std == 0:
                    continue

                query_norm = (query_interp - np.mean(query_interp)) / query_std
                db_norm = (db_interp - np.mean(db_interp)) / db_std

                correlation = np.corrcoef(query_norm, db_norm)[0, 1]
                window_similarity = abs(correlation)

                total_weighted_score += window_similarity * window["weight"]
                total_weight += window["weight"]

            if total_weight == 0:
                return 0.0

            final_score = total_weighted_score / total_weight
            return max(0, min(1, final_score))

        except Exception as e:
            print(f"Error in multi-window score calculation: {e}")
            return 0.0

    def calculate_mineral_vibration_score(self, db_wavenumbers, db_intensities):
        """Calculate correlation score weighted by mineral vibrational mode significance."""
        try:
            db_wavenumbers = np.array(db_wavenumbers, dtype=float)
            db_intensities = np.array(db_intensities, dtype=float)

            if len(db_wavenumbers) == 0 or len(db_intensities) == 0:
                return 0.0

            vibrational_modes = [
                {"name": "Metal-O lattice", "range": (50, 300), "weight": 0.15},
                {"name": "M-O bending", "range": (300, 500), "weight": 0.20},
                {"name": "Ring/chain bending", "range": (500, 700), "weight": 0.25},
                {"name": "Si-O stretching", "range": (700, 1000), "weight": 0.30},
                {"name": "CO3/SO4 modes", "range": (1000, 1200), "weight": 0.10}
            ]

            total_weighted_score = 0.0
            total_weight = 0.0

            for mode in vibrational_modes:
                mode_start, mode_end = mode["range"]

                query_mask = (self.current_wavenumbers >= mode_start) & (self.current_wavenumbers <= mode_end)
                db_mask = (db_wavenumbers >= mode_start) & (db_wavenumbers <= mode_end)

                if not np.any(query_mask) or not np.any(db_mask):
                    continue

                overlap_start = max(mode_start,
                                  max(self.current_wavenumbers[query_mask].min(), db_wavenumbers[db_mask].min()))
                overlap_end = min(mode_end,
                                min(self.current_wavenumbers[query_mask].max(), db_wavenumbers[db_mask].max()))

                if overlap_end <= overlap_start:
                    continue

                base_points = 30 if mode["weight"] >= 0.25 else 20
                n_points = min(base_points, np.sum(query_mask), np.sum(db_mask))

                if n_points < 3:
                    continue

                common_wavenumbers = np.linspace(overlap_start, overlap_end, n_points)

                query_interp = np.interp(common_wavenumbers, self.current_wavenumbers, self.processed_intensities)
                db_interp = np.interp(common_wavenumbers, db_wavenumbers, db_intensities)

                peak_intensity_weight = np.mean(query_interp) / np.max(self.processed_intensities) if np.max(self.processed_intensities) > 0 else 0.1
                effective_weight = mode["weight"] * (1 + peak_intensity_weight)

                query_std = np.std(query_interp)
                db_std = np.std(db_interp)

                if query_std == 0 or db_std == 0:
                    continue

                query_norm = (query_interp - np.mean(query_interp)) / query_std
                db_norm = (db_interp - np.mean(db_interp)) / db_std

                correlation = np.corrcoef(query_norm, db_norm)[0, 1]
                mode_similarity = abs(correlation)

                total_weighted_score += mode_similarity * effective_weight
                total_weight += effective_weight

            if total_weight == 0:
                return 0.0

            final_score = total_weighted_score / total_weight
            return max(0, min(1, final_score))

        except Exception as e:
            print(f"Error in mineral vibration score calculation: {e}")
            return 0.0

    def calculate_dtw_score(self, db_wavenumbers, db_intensities):
        """Calculate optimized DTW similarity score between current spectrum and database spectrum."""
        try:
            from fastdtw import fastdtw

            db_wavenumbers = np.array(db_wavenumbers, dtype=float)
            db_intensities = np.array(db_intensities, dtype=float)

            if len(db_wavenumbers) == 0 or len(db_intensities) == 0:
                return 0.0

            overlap_start = max(self.current_wavenumbers.min(), db_wavenumbers.min())
            overlap_end = min(self.current_wavenumbers.max(), db_wavenumbers.max())

            query_range = self.current_wavenumbers.max() - self.current_wavenumbers.min()
            db_range = db_wavenumbers.max() - db_wavenumbers.min()
            overlap_range = overlap_end - overlap_start

            if overlap_range < 0.3 * min(query_range, db_range):
                return 0.0

            n_points = min(75, len(self.current_wavenumbers), len(db_wavenumbers))
            common_wavenumbers = np.linspace(overlap_start, overlap_end, n_points)

            query_interp = np.interp(common_wavenumbers, self.current_wavenumbers, self.processed_intensities)
            db_interp = np.interp(common_wavenumbers, db_wavenumbers, db_intensities)

            if np.std(query_interp) == 0 or np.std(db_interp) == 0:
                return 0.0

            query_norm = (query_interp - query_interp.min()) / (query_interp.max() - query_interp.min())
            db_norm = (db_interp - db_interp.min()) / (db_interp.max() - db_interp.min())

            distance, path = fastdtw(query_norm, db_norm, radius=min(10, n_points//5))

            if len(path) > 0:
                normalized_distance = distance / len(path)
                similarity = 1.0 / (1.0 + normalized_distance * 3.0)
                return max(0, min(1, similarity))
            else:
                return 0.0

        except ImportError:
            return self.calculate_correlation_score(db_wavenumbers, db_intensities)
        except Exception as e:
            print(f"Error in DTW score calculation: {e}")
            return 0.0

    def calculate_peak_score(self, db_wavenumbers, db_intensities, user_specified_peaks=None, peak_tolerance=10):
        """Calculate peak-based similarity score between current spectrum and database spectrum."""
        try:
            db_wavenumbers = np.array(db_wavenumbers, dtype=float)
            db_intensities = np.array(db_intensities, dtype=float)

            if len(db_wavenumbers) == 0 or len(db_intensities) == 0:
                return 0.0

            db_peaks, _ = find_peaks(db_intensities, height=np.max(db_intensities) * 0.1)

            if len(db_peaks) == 0:
                return 0.0

            db_peak_positions = db_wavenumbers[db_peaks]
            db_peak_intensities = db_intensities[db_peaks]

            if user_specified_peaks and len(user_specified_peaks) > 0:
                query_peak_positions = np.array(user_specified_peaks)
                if self.current_wavenumbers is not None and self.processed_intensities is not None:
                    query_peak_intensities = np.interp(query_peak_positions, self.current_wavenumbers, self.processed_intensities)
                else:
                    query_peak_intensities = np.ones_like(query_peak_positions)

                score_type = "user_specified"
            else:
                if self.detected_peaks is None or len(self.detected_peaks) == 0:
                    return 0.0

                query_peak_positions = self.current_wavenumbers[self.detected_peaks]
                query_peak_intensities = self.processed_intensities[self.detected_peaks]
                score_type = "detected_peaks"

            matched_peaks = 0
            total_intensity_diff = 0
            peak_matches = []

            for i, (query_pos, query_int) in enumerate(zip(query_peak_positions, query_peak_intensities)):
                distances = np.abs(db_peak_positions - query_pos)
                min_dist_idx = np.argmin(distances)
                min_distance = distances[min_dist_idx]

                if min_distance <= peak_tolerance:
                    matched_peaks += 1
                    peak_matches.append({
                        'query_pos': query_pos,
                        'db_pos': db_peak_positions[min_dist_idx],
                        'distance': min_distance
                    })

                    db_int = db_peak_intensities[min_dist_idx]

                    if score_type == "user_specified":
                        query_norm = query_int / np.max(query_peak_intensities) if np.max(query_peak_intensities) > 0 else 1
                        db_norm = db_int / np.max(db_peak_intensities) if np.max(db_peak_intensities) > 0 else 1
                        intensity_diff = abs(query_norm - db_norm) * 0.3
                    else:
                        query_norm = query_int / np.max(query_peak_intensities)
                        db_norm = db_int / np.max(db_peak_intensities)
                        intensity_diff = abs(query_norm - db_norm)

                    total_intensity_diff += intensity_diff

            if matched_peaks == 0:
                return 0.0

            peak_match_ratio = matched_peaks / len(query_peak_positions)
            avg_intensity_similarity = 1 - (total_intensity_diff / matched_peaks)

            if score_type == "user_specified":
                score = 0.8 * peak_match_ratio + 0.2 * avg_intensity_similarity

                if peak_match_ratio >= 0.8:
                    score = min(1.0, score * 1.1)

            else:
                score = 0.7 * peak_match_ratio + 0.3 * avg_intensity_similarity

            return max(0, min(1, score))

        except Exception as e:
            print(f"Error in peak score calculation: {e}")
            return 0.0

    def calculate_combined_score(self, db_wavenumbers, db_intensities, user_specified_peaks=None, peak_tolerance=10):
        """Calculate optimized combined similarity score using correlation, DTW, and optionally peak matching."""
        try:
            correlation_score = self.calculate_correlation_score(db_wavenumbers, db_intensities)

            if correlation_score < 0.2:
                return correlation_score * 0.5

            if correlation_score < 0.5:
                dtw_score = self.calculate_dtw_score(db_wavenumbers, db_intensities)
                if user_specified_peaks and len(user_specified_peaks) > 0:
                    peak_score = self.calculate_peak_score(db_wavenumbers, db_intensities, user_specified_peaks, peak_tolerance)
                    combined_score = 0.4 * correlation_score + 0.3 * dtw_score + 0.3 * peak_score
                else:
                    combined_score = 0.6 * correlation_score + 0.4 * dtw_score
            else:
                dtw_score = self.calculate_dtw_score(db_wavenumbers, db_intensities)

                if user_specified_peaks and len(user_specified_peaks) > 0:
                    peak_score = self.calculate_peak_score(db_wavenumbers, db_intensities, user_specified_peaks, peak_tolerance)
                    combined_score = 0.2 * correlation_score + 0.3 * dtw_score + 0.5 * peak_score
                else:
                    combined_score = 0.3 * correlation_score + 0.7 * dtw_score

            return max(0, min(1, combined_score))

        except Exception as e:
            print(f"Error in combined score calculation: {e}")
            return 0.0
