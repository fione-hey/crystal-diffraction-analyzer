import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from src.diffraction_analyzer import CrystalDiffractionAnalyzer

def test_basic_functionality():
    print("Testing basic functionality...")
    image = np.zeros((100, 100))
    image[50, 50] = 1.0
    
    analyzer = CrystalDiffractionAnalyzer()
    peaks, intensities, info = analyzer.find_2d_peaks_robust(image)
    
    print(f"Found {len(peaks)} peaks in test image")
    return len(peaks) > 0

if __name__ == "__main__":
    success = test_basic_functionality()
    if success:
        print("All tests passed!")
    else:
        print("Tests failed!")