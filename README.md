# Crystal Diffraction Analyzer

Advanced 2D crystal diffraction peak detection and analysis with Monte Carlo uncertainty quantification.  
Developed for crystal diffraction analysis at ORNL SSN Diffraction Section.

## Features

- Robust 2D peak detection using advanced blob detection
- Smart noise filtering and multi-hump peak grouping  
- 2D to 3D reciprocal space mapping for Miller index assignment
- Monte Carlo uncertainty quantification with bootstrap analysis
- Crystal symmetry analysis and lattice parameter estimation
- Comprehensive visualization and error reporting

## Installation

### Method 1: Download ZIP (Easiest)

1. Click the green "Code" button above
2. Click "Download ZIP" 
3. Unzip the file on your computer
4. Follow setup steps below

### Method 2: Git Clone

```bash
git clone https://github.com/ilovethis48/crystal-diffraction-analyzer.git
cd crystal-diffraction-analyzer
```

## Setup Steps (Required for Both Methods)

```bash
# Create virtual environment
python -m venv diffraction_env

# Activate environment
# Windows:
diffraction_env\Scripts\activate

# Mac/Linux:
source diffraction_env/bin/activate

# Install required packages
pip install -r requirements.txt

# Test installation
python tests/test_analyzer.py
```

## Quick Start

### Run Demo

```bash
python examples/demo_analysis.py
```

### Analyze Your Own Data

```python
from src.diffraction_analyzer import CrystalDiffractionAnalyzer

# Load your diffraction image
analyzer = CrystalDiffractionAnalyzer()
peaks, intensities, info = analyzer.find_2d_peaks_robust(your_image)
```

### For Real Detector Data

```python
# Set up your detector parameters
detector_params = {
    'distance': 165,        # mm - detector distance
    'pixel_size': 0.079,    # mm - pixel size
    'beam_center': [1024, 1024],  # pixels - beam center
    'wavelength': 1.54      # Angstrom - X-ray wavelength
}

# Analyze with reciprocal space mapping
q_vectors = analyzer.reciprocal_space_mapping(peaks, detector_params)
```

## Examples

- `examples/demo_analysis.py` - Full demonstration
- `tests/test_analyzer.py` - Verification tests

## Requirements

- Python 3.9+
- numpy, scipy, matplotlib, scikit-image, scikit-learn, pandas, pillow

## Author

Created by [Fiona He] - ORNL SSN Diffraction Section Intern
