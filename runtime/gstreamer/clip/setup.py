from setuptools import setup, find_packages
import subprocess
import os

# Read requirements.txt
with open("requirements.txt", "r") as f:
    requirements = f.read().splitlines()

# Compile C++ code using Meson
try:
    subprocess.run(["./compile_postprocess.sh"])
except Exception as e:
    print(f"Failed to compile C++ code: {e}")

# download hef files if not exist
# check if hef files exist on the resources folder
if not os.path.isfile("resources/yolov5s_personface.hef"):
    print("Downloading hef files...")
    subprocess.run(["./download_hef.sh"])
    
# Setup function
setup(
    name='clip_app',
    version='0.1',
    packages=find_packages(),
    install_requires=requirements,
    scripts=['compile_postprocess.sh'],
    entry_points={
        'console_scripts': [
            'clip_app = clip_app:main',
        ],
    },
)