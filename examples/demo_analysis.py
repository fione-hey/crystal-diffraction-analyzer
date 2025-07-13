import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.diffraction_analyzer import demo_crystal_diffraction, demo_monte_carlo_diffraction

def main():
    print("Crystal Diffraction Analyzer Demo")
    print("Running basic analysis...")
    demo_crystal_diffraction()
    
    print("Running Monte Carlo analysis...")
    demo_monte_carlo_diffraction()

if __name__ == "__main__":
    main()