import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, ndimage
from scipy.optimize import curve_fit
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from skimage import filters, feature, measure, morphology, segmentation
from skimage.restoration import denoise_bilateral
import warnings
warnings.filterwarnings('ignore')

class CrystalDiffractionAnalyzer:
    def __init__(self, noise_threshold=0.1, min_peak_area=10, peak_merge_distance=5):
        """
        2D Crystal Diffraction Peak Analyzer with reciprocal space mapping
        
        Parameters:
        - noise_threshold: Relative threshold for noise filtering
        - min_peak_area: Minimum area (pixels) for a valid diffraction spot
        - peak_merge_distance: Distance threshold for merging nearby peaks
        """
        self.noise_threshold = noise_threshold
        self.min_peak_area = min_peak_area
        self.peak_merge_distance = peak_merge_distance
        
    def preprocess_2d_data(self, image, sigma=1.0):
        """
        Preprocess 2D diffraction image to enhance peaks and reduce noise
        """
        # Use bilateral denoising to preserve sharp diffraction spots
        denoised = denoise_bilateral(image, sigma_color=0.1, sigma_spatial=sigma)
        
        # Apply Gaussian filter for additional smoothing if needed
        if sigma > 0:
            smoothed = filters.gaussian(denoised, sigma=sigma, preserve_range=True)
        else:
            smoothed = denoised
            
        return smoothed
    
    def find_2d_peaks_robust(self, image):
        """
        Find diffraction peaks in 2D detector image using advanced blob detection
        """
        # Preprocess the image
        processed_image = self.preprocess_2d_data(image)
        
        # Estimate background and noise level
        background = filters.rank.median(processed_image.astype(np.uint16), 
                                       morphology.disk(10))
        background_subtracted = processed_image - background
        background_subtracted = np.maximum(background_subtracted, 0)
        
        # Estimate noise level using robust statistics
        noise_level = np.median(np.abs(background_subtracted - np.median(background_subtracted))) * 1.4826
        
        # Dynamic threshold based on image statistics
        threshold = np.percentile(background_subtracted, 95) * self.noise_threshold
        threshold = max(threshold, noise_level * 3)
        
        # Use multiple blob detection methods for robustness
        peaks_dog = feature.blob_dog(background_subtracted, 
                                   min_sigma=0.5, max_sigma=5, 
                                   threshold=threshold/np.max(background_subtracted))
        
        peaks_log = feature.blob_log(background_subtracted,
                                   min_sigma=0.5, max_sigma=5,
                                   threshold=threshold/np.max(background_subtracted))
        
        # Combine and filter peaks
        all_peaks = []
        if len(peaks_dog) > 0:
            all_peaks.extend(peaks_dog)
        if len(peaks_log) > 0:
            all_peaks.extend(peaks_log)
        
        if len(all_peaks) == 0:
            return np.array([]), {}
        
        all_peaks = np.array(all_peaks)
        
        # Remove duplicate peaks that are too close
        if len(all_peaks) > 1:
            unique_peaks = self.remove_duplicate_peaks(all_peaks[:, :2])  # Only x,y coordinates
            # Get intensities at peak positions
            peak_intensities = []
            for peak in unique_peaks:
                y_idx, x_idx = int(peak[0]), int(peak[1])
                if 0 <= y_idx < image.shape[0] and 0 <= x_idx < image.shape[1]:
                    peak_intensities.append(background_subtracted[y_idx, x_idx])
                else:
                    peak_intensities.append(0)
        else:
            unique_peaks = all_peaks[:, :2]
            peak_intensities = [background_subtracted[int(unique_peaks[0][0]), int(unique_peaks[0][1])]]
        
        # Filter peaks by minimum area and intensity
        valid_peaks = []
        valid_intensities = []
        
        for i, (peak, intensity) in enumerate(zip(unique_peaks, peak_intensities)):
            # Check if peak meets minimum criteria
            if intensity > threshold:
                # Calculate local peak area using watershed
                area = self.calculate_peak_area(background_subtracted, peak, threshold)
                if area >= self.min_peak_area:
                    valid_peaks.append(peak)
                    valid_intensities.append(intensity)
        
        peak_info = {
            'raw_image': image,
            'processed_image': processed_image,
            'background_subtracted': background_subtracted,
            'background': background,
            'noise_level': noise_level,
            'threshold': threshold,
            'num_raw_peaks': len(all_peaks),
            'num_valid_peaks': len(valid_peaks)
        }
        
        return np.array(valid_peaks), np.array(valid_intensities), peak_info
    
    def remove_duplicate_peaks(self, peaks):
        """
        Remove peaks that are too close to each other
        """
        if len(peaks) <= 1:
            return peaks
        
        # Use DBSCAN clustering to group nearby peaks
        clustering = DBSCAN(eps=self.peak_merge_distance, min_samples=1)
        clusters = clustering.fit_predict(peaks)
        
        # For each cluster, keep the peak with highest intensity
        unique_peaks = []
        for cluster_id in np.unique(clusters):
            cluster_peaks = peaks[clusters == cluster_id]
            if len(cluster_peaks) == 1:
                unique_peaks.append(cluster_peaks[0])
            else:
                # For now, just take the centroid - you might want to use intensity weighting
                centroid = np.mean(cluster_peaks, axis=0)
                unique_peaks.append(centroid)
        
        return np.array(unique_peaks)
    
    def calculate_peak_area(self, image, peak_center, threshold):
        """
        Calculate the area of a diffraction spot using watershed segmentation
        """
        y_center, x_center = int(peak_center[0]), int(peak_center[1])
        
        # Define a local region around the peak
        margin = 10
        y_min = max(0, y_center - margin)
        y_max = min(image.shape[0], y_center + margin)
        x_min = max(0, x_center - margin)
        x_max = min(image.shape[1], x_center + margin)
        
        local_region = image[y_min:y_max, x_min:x_max]
        
        # Simple thresholding for area calculation
        binary_region = local_region > threshold * 0.5
        area = np.sum(binary_region)
        
        return area
    
    def reciprocal_space_mapping(self, peak_positions, detector_params):
        """
        Convert 2D detector coordinates to 3D reciprocal space coordinates
        
        Parameters:
        detector_params: dict with keys:
        - 'distance': detector distance (mm)
        - 'pixel_size': pixel size (mm)
        - 'beam_center': [x_center, y_center] in pixels
        - 'wavelength': X-ray wavelength (Angstroms)
        - 'detector_normal': [nx, ny, nz] detector normal vector
        """
        if len(peak_positions) == 0:
            return np.array([])
        
        # Extract parameters
        distance = detector_params['distance']
        pixel_size = detector_params['pixel_size']
        beam_center = detector_params['beam_center']
        wavelength = detector_params['wavelength']
        
        # Convert pixel coordinates to real detector coordinates (mm)
        x_det = (peak_positions[:, 1] - beam_center[0]) * pixel_size
        y_det = (peak_positions[:, 0] - beam_center[1]) * pixel_size
        z_det = np.full_like(x_det, distance)
        
        # Calculate scattering vector length
        detector_distance = np.sqrt(x_det**2 + y_det**2 + z_det**2)
        
        # Calculate scattering angles
        two_theta = np.arctan(np.sqrt(x_det**2 + y_det**2) / distance)
        
        # Calculate reciprocal space coordinates (q-space)
        # |q| = 4π sin(θ) / λ
        q_magnitude = 4 * np.pi * np.sin(two_theta / 2) / wavelength
        
        # Direction cosines for q-vector
        q_x = q_magnitude * x_det / detector_distance
        q_y = q_magnitude * y_det / detector_distance
        q_z = q_magnitude * (distance / detector_distance - 1)  # Assuming incident beam along z
        
        return np.column_stack([q_x, q_y, q_z])
    
    def analyze_crystal_symmetry(self, q_vectors):
        """
        Analyze potential crystal symmetry from reciprocal space vectors
        """
        if len(q_vectors) < 3:
            return {"symmetry": "insufficient_data", "analysis": "Need at least 3 peaks"}
        
        # Calculate distances from origin
        q_magnitudes = np.linalg.norm(q_vectors, axis=1)
        
        # Look for systematic relationships
        analysis = {}
        
        # Check for cubic symmetry (equal spacings)
        unique_magnitudes = np.unique(np.round(q_magnitudes, decimals=3))
        if len(unique_magnitudes) < len(q_magnitudes):
            analysis['repeated_spacings'] = True
            analysis['unique_q_values'] = unique_magnitudes
        else:
            analysis['repeated_spacings'] = False
        
        # Check for orthogonal relationships
        if len(q_vectors) >= 3:
            # Calculate angles between q-vectors
            angles = []
            for i in range(len(q_vectors)):
                for j in range(i+1, len(q_vectors)):
                    cos_angle = np.dot(q_vectors[i], q_vectors[j]) / (
                        np.linalg.norm(q_vectors[i]) * np.linalg.norm(q_vectors[j]))
                    # Clamp to valid range for arccos
                    cos_angle = np.clip(cos_angle, -1, 1)
                    angle = np.arccos(cos_angle) * 180 / np.pi
                    angles.append(angle)
            
            # Look for 90-degree angles (orthogonal system)
            right_angles = np.sum(np.abs(np.array(angles) - 90) < 5)  # 5-degree tolerance
            analysis['orthogonal_angles'] = right_angles
            analysis['all_angles'] = angles
        
        # Estimate lattice parameters (simplified)
        if analysis.get('repeated_spacings', False):
            # For cubic: a = 2π/q
            a_estimates = 2 * np.pi / unique_magnitudes
            analysis['lattice_parameter_estimates'] = a_estimates
        
        return analysis
    
    def visualize_2d_analysis(self, image, peaks, intensities, peak_info, q_vectors=None):
        """
        Comprehensive visualization for 2D diffraction analysis
        """
        # Create figure with better spacing
        fig = plt.figure(figsize=(18, 14))
        
        # Use gridspec for better control over spacing
        from matplotlib.gridspec import GridSpec
        gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3, 
                     left=0.08, right=0.95, top=0.93, bottom=0.08)
        
        # Plot 1: Original image with detected peaks
        ax1 = fig.add_subplot(gs[0, 0])
        im1 = ax1.imshow(image, cmap='viridis', origin='lower')
        if len(peaks) > 0:
            ax1.plot(peaks[:, 1], peaks[:, 0], 'ro', markersize=6, markerfacecolor='none', 
                    markeredgewidth=2, label=f'{len(peaks)} peaks')
            # Add peak numbers (smaller, less crowded)
            for i, peak in enumerate(peaks):
                if i < 15:  # Only label first 15 peaks to avoid clutter
                    ax1.annotate(str(i), (peak[1], peak[0]), xytext=(3, 3), 
                               textcoords='offset points', color='red', fontsize=8, fontweight='bold')
        
        ax1.set_title('Original Image + Detected Peaks', fontsize=12, pad=10)
        ax1.set_xlabel('X (pixels)', fontsize=10)
        ax1.set_ylabel('Y (pixels)', fontsize=10)
        if len(peaks) > 0:
            ax1.legend(fontsize=9)
        
        # Colorbar with proper spacing
        cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.8)
        cbar1.ax.tick_params(labelsize=8)
        
        # Plot 2: Background subtracted image
        ax2 = fig.add_subplot(gs[0, 1])
        im2 = ax2.imshow(peak_info['background_subtracted'], cmap='viridis', origin='lower')
        ax2.set_title('Background Subtracted', fontsize=12, pad=10)
        ax2.set_xlabel('X (pixels)', fontsize=10)
        ax2.set_ylabel('Y (pixels)', fontsize=10)
        
        cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.8)
        cbar2.ax.tick_params(labelsize=8)
        
        # Plot 3: Peak intensity distribution
        ax3 = fig.add_subplot(gs[0, 2])
        if len(intensities) > 0:
            bars = ax3.bar(range(len(intensities)), intensities, alpha=0.7, width=0.8)
            ax3.set_xlabel('Peak Number', fontsize=10)
            ax3.set_ylabel('Intensity', fontsize=10)
            ax3.set_title('Peak Intensities', fontsize=12, pad=10)
            ax3.grid(True, alpha=0.3)
            ax3.tick_params(labelsize=8)
            
            # Color bars by intensity
            max_int = np.max(intensities)
            for bar, intensity in zip(bars, intensities):
                bar.set_color(plt.cm.plasma(intensity / max_int))
        else:
            ax3.text(0.5, 0.5, 'No peaks detected', ha='center', va='center', 
                    transform=ax3.transAxes, fontsize=12)
            ax3.set_title('Peak Intensities', fontsize=12, pad=10)
        
        # Plot 4: Radial intensity profile
        ax4 = fig.add_subplot(gs[1, 0])
        center_y, center_x = np.array(image.shape) // 2
        y, x = np.ogrid[:image.shape[0], :image.shape[1]]
        r = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        
        # Create radial bins
        r_max = min(center_x, center_y)
        r_bins = np.linspace(0, r_max, 50)
        radial_profile = []
        
        for i in range(len(r_bins)-1):
            mask = (r >= r_bins[i]) & (r < r_bins[i+1])
            if np.any(mask):
                radial_profile.append(np.mean(image[mask]))
            else:
                radial_profile.append(0)
        
        ax4.plot(r_bins[:-1], radial_profile, 'b-', linewidth=2)
        ax4.set_xlabel('Radial Distance (pixels)', fontsize=10)
        ax4.set_ylabel('Average Intensity', fontsize=10)
        ax4.set_title('Radial Intensity Profile', fontsize=12, pad=10)
        ax4.grid(True, alpha=0.3)
        ax4.tick_params(labelsize=8)
        
        # Plot 5: Reciprocal space map (if q_vectors provided)
        ax5 = fig.add_subplot(gs[1, 1])
        if q_vectors is not None and len(q_vectors) > 0:
            # 2D projection of reciprocal space
            scatter = ax5.scatter(q_vectors[:, 0], q_vectors[:, 1], 
                                c=intensities if len(intensities) > 0 else 'blue', 
                                s=80, alpha=0.7, cmap='plasma')
            
            # Add origin
            ax5.plot(0, 0, 'k+', markersize=12, markeredgewidth=2, label='Origin')
            
            # Add peak labels (only first 10 to avoid clutter)
            for i, q in enumerate(q_vectors[:10]):
                ax5.annotate(str(i), (q[0], q[1]), xytext=(3, 3), 
                           textcoords='offset points', fontsize=8, fontweight='bold')
            
            ax5.set_xlabel('qx (Å⁻¹)', fontsize=10)
            ax5.set_ylabel('qy (Å⁻¹)', fontsize=10)
            ax5.set_title('Reciprocal Space Map (qx-qy)', fontsize=12, pad=10)
            ax5.grid(True, alpha=0.3)
            ax5.legend(fontsize=9)
            ax5.axis('equal')
            ax5.tick_params(labelsize=8)
        else:
            ax5.text(0.5, 0.5, 'Reciprocal space mapping\nrequires detector parameters', 
                    ha='center', va='center', transform=ax5.transAxes, fontsize=10)
            ax5.set_title('Reciprocal Space Map', fontsize=12, pad=10)
        
        # Plot 6: Analysis summary
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.axis('off')
        
        # Create summary text with better formatting
        summary_lines = [
            "ANALYSIS SUMMARY",
            "=" * 20,
            f"Total peaks found: {len(peaks)}",
            f"Noise level: {peak_info['noise_level']:.4f}",
            f"Detection threshold: {peak_info['threshold']:.4f}",
            "",
            f"Image dimensions: {image.shape}",
            f"Background level: {np.mean(peak_info['background']):.2f}",
            "",
            "Peak Statistics:"
        ]
        
        if len(intensities) > 0:
            summary_lines.extend([
                f"- Max intensity: {np.max(intensities):.2f}",
                f"- Mean intensity: {np.mean(intensities):.2f}",
                f"- Std intensity: {np.std(intensities):.2f}"
            ])
        
        if q_vectors is not None and len(q_vectors) > 0:
            q_mags = np.linalg.norm(q_vectors, axis=1)
            summary_lines.extend([
                "",
                "Reciprocal Space:",
                f"- q-range: {np.min(q_mags):.3f} - {np.max(q_mags):.3f} Å⁻¹",
                f"- d-spacing range: {2*np.pi/np.max(q_mags):.2f} - {2*np.pi/np.min(q_mags):.2f} Å"
            ])
        
        # Display text with proper spacing
        summary_text = '\n'.join(summary_lines)
        ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, 
                fontfamily='monospace', fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5))
        
        ax6.set_title('Summary', fontsize=12, pad=10)
        
        return fig

# Example usage for crystal diffraction
def demo_crystal_diffraction():
    """
    Demonstrate crystal diffraction analysis with synthetic 2D data
    """
    # Create synthetic 2D diffraction pattern
    image_size = 512
    image = np.zeros((image_size, image_size))
    
    # Add realistic diffraction spots (simulating crystal diffraction pattern)
    spot_positions = [
        (150, 200), (200, 150), (300, 280), (180, 350),
        (400, 200), (350, 120), (450, 400), (120, 300),
        (380, 380), (250, 100), (100, 250), (320, 160)
    ]
    
    spot_intensities = [1.0, 0.8, 0.9, 0.6, 0.7, 0.5, 0.4, 0.8, 0.3, 0.9, 0.6, 0.7]
    spot_widths = [3, 2.5, 4, 2, 3.5, 2, 2.5, 3, 2, 4, 2.5, 3]
    
    # Create Gaussian spots
    y_grid, x_grid = np.ogrid[:image_size, :image_size]
    
    for (y_pos, x_pos), intensity, width in zip(spot_positions, spot_intensities, spot_widths):
        spot = intensity * np.exp(-((x_grid - x_pos)**2 + (y_grid - y_pos)**2) / (2 * width**2))
        image += spot
    
    # Add some multi-peak clusters (realistic peak splitting)
    # Cluster 1: Close peaks that should be grouped
    image += 0.4 * np.exp(-((x_grid - 220)**2 + (y_grid - 220)**2) / (2 * 2**2))
    image += 0.3 * np.exp(-((x_grid - 225)**2 + (y_grid - 223)**2) / (2 * 2**2))
    
    # Add background and noise
    background = 0.1 + 0.0001 * (x_grid + y_grid)  # Gradual background
    noise = np.random.normal(0, 0.05, (image_size, image_size))
    image_with_noise = image + background + noise
    
    # Ensure non-negative values
    image_with_noise = np.maximum(image_with_noise, 0)
    
    # Create analyzer
    analyzer = CrystalDiffractionAnalyzer(
        noise_threshold=0.15,
        min_peak_area=8,
        peak_merge_distance=10
    )
    
    # Find peaks
    peaks, intensities, peak_info = analyzer.find_2d_peaks_robust(image_with_noise)
    print(f"Found {len(peaks)} diffraction spots")
    
    # Example detector parameters for reciprocal space mapping
    detector_params = {
        'distance': 100,  # mm
        'pixel_size': 0.1,  # mm per pixel
        'beam_center': [image_size//2, image_size//2],  # center of detector
        'wavelength': 1.54,  # Angstroms (Cu K-alpha)
        'detector_normal': [0, 0, 1]  # detector normal along z-axis
    }
    
    # Convert to reciprocal space
    q_vectors = analyzer.reciprocal_space_mapping(peaks, detector_params)
    
    # Analyze crystal symmetry
    if len(q_vectors) > 0:
        symmetry_analysis = analyzer.analyze_crystal_symmetry(q_vectors)
        print("Crystal symmetry analysis:")
        for key, value in symmetry_analysis.items():
            print(f"  {key}: {value}")
    
    # Create comprehensive visualization
    fig = analyzer.visualize_2d_analysis(image_with_noise, peaks, intensities, peak_info, q_vectors)
    
    # Adjust layout to prevent overlapping
    plt.suptitle('Crystal Diffraction Analysis Results', fontsize=16, y=0.98)
    plt.show()
    
    # Print reciprocal space coordinates
    if len(q_vectors) > 0:
        print("\nReciprocal space coordinates (qx, qy, qz) in Å⁻¹:")
        for i, q in enumerate(q_vectors):
            d_spacing = 2 * np.pi / np.linalg.norm(q)
            print(f"Peak {i}: q = ({q[0]:.4f}, {q[1]:.4f}, {q[2]:.4f}), d = {d_spacing:.2f} Å")
    
    return analyzer, peaks, intensities, peak_info, q_vectors

# Utility functions for real data analysis
def load_and_analyze_real_data(image_path, detector_params):
    """
    Load and analyze real diffraction data
    Usage example:
    
    detector_params = {
        'distance': 165,  # mm - your detector distance
        'pixel_size': 0.079,  # mm - your pixel size  
        'beam_center': [1024, 1024],  # pixels - beam center
        'wavelength': 1.54,  # Angstroms - your wavelength
    }
    
    analyzer, peaks, intensities, peak_info, q_vectors = load_and_analyze_real_data(
        'your_diffraction_image.tif', detector_params)
    """
    from skimage import io
    
    # Load image
    image = io.imread(image_path)
    
    # Convert to float if needed
    if image.dtype == np.uint16 or image.dtype == np.uint8:
        image = image.astype(np.float32)
    
    # Create analyzer
    analyzer = CrystalDiffractionAnalyzer()
    
    # Find peaks
    peaks, intensities, peak_info = analyzer.find_2d_peaks_robust(image)
    
    # Convert to reciprocal space
    q_vectors = analyzer.reciprocal_space_mapping(peaks, detector_params)
    
    # Analyze symmetry
    symmetry_analysis = analyzer.analyze_crystal_symmetry(q_vectors)
    
    # Visualize
    fig = analyzer.visualize_2d_analysis(image, peaks, intensities, peak_info, q_vectors)
    
    return analyzer, peaks, intensities, peak_info, q_vectors, symmetry_analysis

# Monte Carlo methods for diffraction analysis
class MonteCarloDiffraction:
    def __init__(self, base_analyzer):
        self.analyzer = base_analyzer
        
    def simulate_realistic_pattern(self, crystal_params, detector_params, noise_level=0.05):
        """
        Monte Carlo simulation of realistic diffraction patterns
        """
        # Extract parameters
        lattice_a = crystal_params.get('lattice_a', 4.0)  # Angstroms
        crystal_size = crystal_params.get('crystal_size', 100)  # nm
        mosaicity = crystal_params.get('mosaicity', 0.1)  # degrees
        num_peaks = crystal_params.get('num_peaks', 20)
        
        image_size = detector_params.get('image_size', 512)
        
        # Create base image
        image = np.zeros((image_size, image_size))
        
        # Simulate systematic reflections based on lattice
        beam_center = detector_params['beam_center']
        distance = detector_params['distance']
        pixel_size = detector_params['pixel_size']
        wavelength = detector_params['wavelength']
        
        # Generate Miller indices for allowed reflections
        max_h = int(4 * np.pi * distance * pixel_size / (wavelength * lattice_a * image_size))
        
        peak_positions = []
        peak_intensities = []
        
        for h in range(-max_h, max_h+1):
            for k in range(-max_h, max_h+1):
                for l in range(-max_h, max_h+1):
                    if h == 0 and k == 0 and l == 0:
                        continue
                    
                    # Calculate q-vector for this reflection
                    q_mag = 2 * np.pi * np.sqrt(h**2 + k**2 + l**2) / lattice_a
                    
                    # Check if this reflection is observable
                    theta = np.arcsin(q_mag * wavelength / (4 * np.pi))
                    if theta > np.pi/4:  # Reasonable detection limit
                        continue
                    
                    # Calculate detector position
                    scattering_angle = 2 * theta
                    
                    # Simplified: assume reflection along one direction
                    det_distance = distance / np.cos(scattering_angle)
                    x_det = distance * np.tan(scattering_angle) * (h / np.sqrt(h**2 + k**2 + l**2))
                    y_det = distance * np.tan(scattering_angle) * (k / np.sqrt(h**2 + k**2 + l**2))
                    
                    # Convert to pixel coordinates
                    x_pixel = beam_center[0] + x_det / pixel_size
                    y_pixel = beam_center[1] + y_det / pixel_size
                    
                    # Check if on detector
                    if 0 < x_pixel < image_size and 0 < y_pixel < image_size:
                        # Calculate structure factor (simplified)
                        intensity = np.exp(-(h**2 + k**2 + l**2) / 10)  # Decreasing with order
                        
                        # Add some randomness
                        intensity *= np.random.uniform(0.5, 1.5)
                        
                        if intensity > 0.1 and len(peak_positions) < num_peaks:
                            peak_positions.append([y_pixel, x_pixel])
                            peak_intensities.append(intensity)
        
        # Add peaks to image with realistic shapes
        y_grid, x_grid = np.ogrid[:image_size, :image_size]
        
        for (y_pos, x_pos), intensity in zip(peak_positions, peak_intensities):
            # Add mosaicity (peak broadening)
            width = np.random.normal(2.5, 0.5)  # Random peak width
            width = max(width, 1.0)  # Minimum width
            
            # Add some asymmetry
            width_x = width * np.random.uniform(0.8, 1.2)
            width_y = width * np.random.uniform(0.8, 1.2)
            
            # Create peak
            peak = intensity * np.exp(-((x_grid - x_pos)**2 / (2 * width_x**2) + 
                                      (y_grid - y_pos)**2 / (2 * width_y**2)))
            image += peak
        
        # Add realistic background
        background = 0.1 + 0.0001 * (x_grid + y_grid)
        background += np.random.uniform(0, 0.05, (image_size, image_size))  # Random background variation
        
        # Add noise
        noise = np.random.normal(0, noise_level, (image_size, image_size))
        
        final_image = image + background + noise
        final_image = np.maximum(final_image, 0)  # Ensure non-negative
        
        return final_image, peak_positions, peak_intensities
    
    def bootstrap_uncertainty_analysis(self, image, n_iterations=100):
        """
        Use bootstrap Monte Carlo to estimate uncertainties in peak parameters
        """
        all_peaks = []
        all_intensities = []
        
        # Store original image statistics
        original_noise = np.std(image)
        
        print(f"Running {n_iterations} bootstrap iterations...")
        
        for i in range(n_iterations):
            # Add slightly different noise realizations
            noise_realization = np.random.normal(0, original_noise * 0.1, image.shape)
            noisy_image = image + noise_realization
            noisy_image = np.maximum(noisy_image, 0)
            
            # Find peaks
            peaks, intensities, _ = self.analyzer.find_2d_peaks_robust(noisy_image)
            
            if len(peaks) > 0:
                all_peaks.append(peaks)
                all_intensities.append(intensities)
        
        # Analyze statistics
        return self._analyze_bootstrap_results(all_peaks, all_intensities)
    
    def _analyze_bootstrap_results(self, all_peaks, all_intensities):
        """
        Analyze bootstrap results to get mean positions and uncertainties
        """
        if len(all_peaks) == 0:
            return {"error": "No peaks found in any iteration"}
        
        # Find most common number of peaks
        peak_counts = [len(peaks) for peaks in all_peaks]
        most_common_count = max(set(peak_counts), key=peak_counts.count)
        
        # Filter to iterations with the most common peak count
        filtered_peaks = [peaks for peaks in all_peaks if len(peaks) == most_common_count]
        filtered_intensities = [intens for peaks, intens in zip(all_peaks, all_intensities) 
                              if len(peaks) == most_common_count]
        
        if len(filtered_peaks) < 3:
            return {"error": "Insufficient consistent results"}
        
        # Match peaks across iterations (simple nearest neighbor)
        reference_peaks = filtered_peaks[0]
        matched_peaks = np.zeros((len(filtered_peaks), len(reference_peaks), 2))
        matched_intensities = np.zeros((len(filtered_peaks), len(reference_peaks)))
        
        for i, (peaks, intensities) in enumerate(zip(filtered_peaks, filtered_intensities)):
            # Match each reference peak to closest found peak
            for j, ref_peak in enumerate(reference_peaks):
                distances = np.linalg.norm(peaks - ref_peak, axis=1)
                closest_idx = np.argmin(distances)
                matched_peaks[i, j] = peaks[closest_idx]
                matched_intensities[i, j] = intensities[closest_idx]
        
        # Calculate statistics
        mean_positions = np.mean(matched_peaks, axis=0)
        std_positions = np.std(matched_peaks, axis=0)
        mean_intensities = np.mean(matched_intensities, axis=0)
        std_intensities = np.std(matched_intensities, axis=0)
        
        results = {
            'n_consistent_peaks': most_common_count,
            'n_bootstrap_samples': len(filtered_peaks),
            'mean_positions': mean_positions,
            'position_uncertainties': std_positions,
            'mean_intensities': mean_intensities,
            'intensity_uncertainties': std_intensities
        }
        
        return results
    
    def robustness_test(self, image, noise_levels=None):
        """
        Test how robust peak detection is to different noise levels
        """
        if noise_levels is None:
            noise_levels = np.logspace(-3, -1, 10)  # 0.001 to 0.1
        
        results = []
        
        for noise_level in noise_levels:
            # Run multiple trials at this noise level
            n_trials = 20
            peaks_found = []
            
            for trial in range(n_trials):
                noise = np.random.normal(0, noise_level * np.max(image), image.shape)
                noisy_image = image + noise
                noisy_image = np.maximum(noisy_image, 0)
                
                peaks, intensities, _ = self.analyzer.find_2d_peaks_robust(noisy_image)
                peaks_found.append(len(peaks))
            
            results.append({
                'noise_level': noise_level,
                'mean_peaks': np.mean(peaks_found),
                'std_peaks': np.std(peaks_found),
                'min_peaks': np.min(peaks_found),
                'max_peaks': np.max(peaks_found)
            })
        
        return results
    
    def visualize_monte_carlo_results(self, image, bootstrap_results, robustness_results):
        """
        Visualize Monte Carlo analysis results
        """
        # Create figure with proper spacing
        fig = plt.figure(figsize=(16, 10))
        
        # Use gridspec for better control
        from matplotlib.gridspec import GridSpec
        gs = GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.3, 
                     left=0.08, right=0.95, top=0.92, bottom=0.10)
        
        # Plot 1: Image with uncertainty ellipses
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(image, cmap='viridis', origin='lower')
        
        if 'mean_positions' in bootstrap_results:
            positions = bootstrap_results['mean_positions']
            uncertainties = bootstrap_results['position_uncertainties']
            
            for i, (pos, unc) in enumerate(zip(positions, uncertainties)):
                # Plot mean position
                ax1.plot(pos[1], pos[0], 'ro', markersize=6)
                
                # Plot uncertainty ellipse (2-sigma)
                from matplotlib.patches import Ellipse
                ellipse = Ellipse((pos[1], pos[0]), 
                                4*unc[1], 4*unc[0],  # 2-sigma ellipse
                                alpha=0.3, facecolor='red')
                ax1.add_patch(ellipse)
                
                # Label (only first 10 to avoid clutter)
                if i < 10:
                    ax1.text(pos[1]+3, pos[0]+3, str(i), color='white', fontweight='bold', fontsize=8)
        
        ax1.set_title('Peak Positions with Uncertainties', fontsize=12, pad=10)
        ax1.set_xlabel('X (pixels)', fontsize=10)
        ax1.set_ylabel('Y (pixels)', fontsize=10)
        ax1.tick_params(labelsize=8)
        
        # Plot 2: Position uncertainties
        ax2 = fig.add_subplot(gs[0, 1])
        if 'position_uncertainties' in bootstrap_results:
            uncertainties = bootstrap_results['position_uncertainties']
            unc_magnitudes = np.linalg.norm(uncertainties, axis=1)
            
            bars = ax2.bar(range(len(unc_magnitudes)), unc_magnitudes, alpha=0.7, width=0.8)
            ax2.set_xlabel('Peak Number', fontsize=10)
            ax2.set_ylabel('Position Uncertainty (pixels)', fontsize=10)
            ax2.set_title('Peak Position Uncertainties', fontsize=12, pad=10)
            ax2.grid(True, alpha=0.3)
            ax2.tick_params(labelsize=8)
            
            # Color bars by uncertainty magnitude
            max_unc = np.max(unc_magnitudes)
            for bar, unc in zip(bars, unc_magnitudes):
                bar.set_color(plt.cm.viridis(unc / max_unc))
        else:
            ax2.text(0.5, 0.5, 'No uncertainty data\navailable', ha='center', va='center', 
                    transform=ax2.transAxes, fontsize=12)
            ax2.set_title('Peak Position Uncertainties', fontsize=12, pad=10)
        
        # Plot 3: Intensity uncertainties
        ax3 = fig.add_subplot(gs[1, 0])
        if 'intensity_uncertainties' in bootstrap_results:
            intensities = bootstrap_results['mean_intensities']
            intensity_errors = bootstrap_results['intensity_uncertainties']
            
            ax3.errorbar(range(len(intensities)), intensities, yerr=intensity_errors,
                        fmt='o', capsize=4, alpha=0.7, markersize=5)
            ax3.set_xlabel('Peak Number', fontsize=10)
            ax3.set_ylabel('Intensity', fontsize=10)
            ax3.set_title('Peak Intensities with Error Bars', fontsize=12, pad=10)
            ax3.grid(True, alpha=0.3)
            ax3.tick_params(labelsize=8)
        else:
            ax3.text(0.5, 0.5, 'No intensity data\navailable', ha='center', va='center', 
                    transform=ax3.transAxes, fontsize=12)
            ax3.set_title('Peak Intensities with Error Bars', fontsize=12, pad=10)
        
        # Plot 4: Robustness to noise
        ax4 = fig.add_subplot(gs[1, 1])
        if robustness_results:
            noise_levels = [r['noise_level'] for r in robustness_results]
            mean_peaks = [r['mean_peaks'] for r in robustness_results]
            std_peaks = [r['std_peaks'] for r in robustness_results]
            
            ax4.errorbar(noise_levels, mean_peaks, yerr=std_peaks, 
                        fmt='o-', capsize=3, alpha=0.7, linewidth=2, markersize=4)
            ax4.set_xscale('log')
            ax4.set_xlabel('Noise Level (relative to max intensity)', fontsize=10)
            ax4.set_ylabel('Number of Peaks Detected', fontsize=10)
            ax4.set_title('Peak Detection Robustness', fontsize=12, pad=10)
            ax4.grid(True, alpha=0.3)
            ax4.tick_params(labelsize=8)
        else:
            ax4.text(0.5, 0.5, 'No robustness data\navailable', ha='center', va='center', 
                    transform=ax4.transAxes, fontsize=12)
            ax4.set_title('Peak Detection Robustness', fontsize=12, pad=10)
        
        return fig

def demo_monte_carlo_diffraction():
    """
    Demonstrate Monte Carlo methods in diffraction analysis
    """
    print("=== Monte Carlo Diffraction Analysis Demo ===\n")
    
    # Create base analyzer
    base_analyzer = CrystalDiffractionAnalyzer()
    mc = MonteCarloDiffraction(base_analyzer)
    
    # 1. Simulate realistic diffraction pattern
    print("1. Simulating realistic crystal diffraction pattern...")
    crystal_params = {
        'lattice_a': 4.0,     # Angstrom
        'crystal_size': 100,   # nm
        'mosaicity': 0.1,     # degrees
        'num_peaks': 15
    }
    
    detector_params = {
        'distance': 100,       # mm
        'pixel_size': 0.1,     # mm
        'beam_center': [256, 256],
        'wavelength': 1.54,    # Angstrom
        'image_size': 512
    }
    
    simulated_image, true_positions, true_intensities = mc.simulate_realistic_pattern(
        crystal_params, detector_params, noise_level=0.03)
    
    print(f"   Generated {len(true_positions)} diffraction spots")
    
    # 2. Bootstrap uncertainty analysis
    print("\n2. Running bootstrap uncertainty analysis...")
    bootstrap_results = mc.bootstrap_uncertainty_analysis(simulated_image, n_iterations=50)
    
    if 'mean_positions' in bootstrap_results:
        print(f"   Found {bootstrap_results['n_consistent_peaks']} consistent peaks")
        print(f"   Average position uncertainty: {np.mean(np.linalg.norm(bootstrap_results['position_uncertainties'], axis=1)):.2f} pixels")
    
    # 3. Robustness testing
    print("\n3. Testing robustness to noise...")
    robustness_results = mc.robustness_test(simulated_image)
    
    # 4. Visualize everything
    print("\n4. Creating comprehensive visualization...")
    fig = mc.visualize_monte_carlo_results(simulated_image, bootstrap_results, robustness_results)
    plt.suptitle('Monte Carlo Diffraction Analysis Results', fontsize=16, y=0.96)
    plt.show()
    
    # 5. Summary statistics
    print("\n=== MONTE CARLO ANALYSIS SUMMARY ===")
    print(f"True number of peaks: {len(true_positions)}")
    
    if 'n_consistent_peaks' in bootstrap_results:
        print(f"Consistently detected peaks: {bootstrap_results['n_consistent_peaks']}")
        
        if bootstrap_results['n_consistent_peaks'] > 0:
            avg_pos_uncertainty = np.mean(np.linalg.norm(bootstrap_results['position_uncertainties'], axis=1))
            avg_int_uncertainty = np.mean(bootstrap_results['intensity_uncertainties'])
            print(f"Average position uncertainty: {avg_pos_uncertainty:.3f} ± pixels")
            print(f"Average intensity uncertainty: {avg_int_uncertainty:.3f}")
    
    # Find noise threshold where detection becomes unreliable
    for result in robustness_results:
        if result['mean_peaks'] < len(true_positions) * 0.8:  # 80% detection threshold
            print(f"Detection becomes unreliable above {result['noise_level']:.4f} noise level")
            break
    
    return mc, simulated_image, bootstrap_results, robustness_results

if __name__ == "__main__":
    # Run both demos
    print("Running Crystal Diffraction Demo...")
    demo_crystal_diffraction()
    
    print("\n" + "="*60 + "\n")
    
    print("Running Monte Carlo Analysis Demo...")
    demo_monte_carlo_diffraction()