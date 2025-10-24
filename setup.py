#!/usr/bin/env python3
"""
FunGen Rewrite - Setup Script

Installation:
    pip install -e .  # Development mode (editable)
    pip install .     # Production install

For GPU support:
    pip install .[gpu]

For development tools:
    pip install .[dev]

Full installation:
    pip install .[all]
"""

from pathlib import Path

from setuptools import find_packages, setup

# Read README
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text() if readme_file.exists() else ""

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
if requirements_file.exists():
    requirements = []
    with open(requirements_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                # Skip torch/tensorrt (user installs separately)
                if not any(x in line for x in ["torch", "tensorrt"]):
                    requirements.append(line)
else:
    requirements = [
        "numpy>=1.24.0",
        "opencv-python>=4.8.0",
        "ultralytics>=8.0.0",
        "scipy>=1.11.0",
        "onnxruntime>=1.16.0",
        "matplotlib>=3.8.0",
        "Pillow>=10.0.0",
    ]

setup(
    name="fungen-rewrite",
    version="1.0.0",
    author="FunGen Multi-Agent Team",
    author_email="noreply@example.com",
    description="AI-Powered Funscript Generator with 100+ FPS tracking on RTX 3090",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ack00gar/FunGen-AI-Powered-Funscript-Generator",
    project_urls={
        "Bug Tracker": "https://github.com/ack00gar/FunGen-AI-Powered-Funscript-Generator/issues",
        "Documentation": "https://github.com/ack00gar/FunGen-AI-Powered-Funscript-Generator/wiki",
        "Source Code": "https://github.com/ack00gar/FunGen-AI-Powered-Funscript-Generator",
    },
    packages=find_packages(exclude=["tests", "tests.*", "examples", "docs"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Multimedia :: Video",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.11",
    install_requires=requirements,
    extras_require={
        "gpu": [
            "torch>=2.1.0",
            "torchvision>=0.16.0",
            # Note: TensorRT must be installed separately from NVIDIA
        ],
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "pytest-mock>=3.12.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "mypy>=1.7.0",
            "flake8>=6.1.0",
            "ipython>=8.18.0",
        ],
        "all": [
            "torch>=2.1.0",
            "torchvision>=0.16.0",
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "mypy>=1.7.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "fungen=main:main",
            "fungen-cli=main:run_cli_mode",
            "fungen-gui=main:run_gui_mode",
        ],
    },
    include_package_data=True,
    package_data={
        "core": ["*.json", "*.yaml"],
        "models": ["*.pt", "*.onnx", "*.engine"],
    },
    zip_safe=False,
    keywords=[
        "funscript",
        "video tracking",
        "object detection",
        "YOLO",
        "ByteTrack",
        "computer vision",
        "TensorRT",
        "GPU acceleration",
    ],
)
