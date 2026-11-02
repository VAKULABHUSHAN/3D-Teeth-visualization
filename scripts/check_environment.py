import argparse
import sys
from pathlib import Path
import shutil

# Make sure we don't hit unicode issues on Windows
OK = "[OK]"
FAIL = "[FAIL]"

def check_environment():
    checks = []
    
    print("Environment Check:")
    print("-" * 30)
    
    # Check Python version
    py_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    if sys.version_info >= (3, 10):
        print(f"{'Python (' + py_version + ')':<20} {OK}")
        checks.append(True)
    else:
        print(f"{'Python (' + py_version + ')':<20} {FAIL} (Requires 3.10+)")
        checks.append(False)
        
    # Check packages
    packages = {
        'cv2': 'OpenCV',
        'numpy': 'NumPy',
        'open3d': 'Open3D',
        'PIL': 'Pillow',
        'yaml': 'PyYAML',
        'loguru': 'Loguru',
        'pydantic': 'Pydantic',
        'tqdm': 'tqdm'
    }
    
    for module, name in packages.items():
        try:
            __import__(module)
            print(f"{name:<20} {OK}")
            checks.append(True)
        except ImportError:
            print(f"{name:<20} {FAIL} (Missing)")
            checks.append(False)
            
    # Check COLMAP
    if shutil.which('colmap'):
        print(f"{'COLMAP':<20} {OK}")
        checks.append(True)
    else:
        print(f"{'COLMAP':<20} {FAIL}")
        print("\nWARNING: COLMAP executable not found in PATH.")
        print("Install COLMAP and ensure it is available in PATH to run the reconstruction phase.")
        checks.append(False)
        
    print("-" * 30)
    if all(checks):
        print("All checks passed. System is ready.")
        return 0
    else:
        print("Some checks failed. Please fix the missing dependencies.")
        return 1

if __name__ == "__main__":
    sys.exit(check_environment())
