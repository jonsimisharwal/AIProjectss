'''
Camera Classifier v0.1 Alpha
Copyright (c) NeuralNine

Instagram: @neuralnine
YouTube: NeuralNine
Website: www.neuralnine.com
'''

import sys
import os

# Add error handling for missing app module
try:
    import app
except ImportError as e:
    print(f"Error importing app module: {e}")
    print("Make sure app.py exists in the same directory")
    sys.exit(1)


def main():
    try:
        app.App(window_title="Camera Classifier v0.1 Alpha")
    except Exception as e:
        print(f"Error starting application: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()