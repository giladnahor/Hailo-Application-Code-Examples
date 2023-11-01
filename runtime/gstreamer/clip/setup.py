from setuptools import setup, find_packages
import subprocess

# Read requirements.txt
with open("requirements.txt", "r") as f:
    requirements = f.read().splitlines()

# Compile C++ code using Meson
try:
    subprocess.run(["./compile_postprocess.sh"])
except Exception as e:
    print(f"Failed to compile C++ code: {e}")

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
