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

## Quick Start

```python
from src.diffraction_analyzer import CrystalDiffractionAnalyzer, demo_crystal_diffraction

# Run the demo
demo_crystal_diffraction()

# Analyze your own data
analyzer = CrystalDiffractionAnalyzer()
peaks, intensities, info = analyzer.find_2d_peaks_robust(your_image)