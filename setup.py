import os
from setuptools import setup, find_packages

# --- Helper function to read requirements ---
def parse_requirements(filename):
    """Load requirements from a pip requirements file."""
    with open(filename, 'r', encoding='utf-8') as f:
        lineiter = (line.strip() for line in f)
        return [line for line in lineiter if line and not line.startswith('#')]

# --- Project Metadata ---
NAME = "ai_video_analyzer"
VERSION = "0.1.0"  #
DESCRIPTION = "An offline AI-powered tool for video analysis and summarization."
AUTHOR = "OptimullPrime"  
AUTHOR_EMAIL = "your.email@example.com" # Replace with your email
URL = "https://github.com/optimull-prime/ai-powered-video-analyzer" # Or your fork's URL
PYTHON_REQUIRES = ">=3.9.0" # Specify the minimum Python version required

# --- Find Packages ---
# find_packages() automatically discovers all packages and sub-packages in the 'src' directory.
# 'where="src"' tells setuptools that the packages are located under the 'src' folder.
packages = find_packages(where="src")

# --- Get Dependencies ---
# Read dependencies from your requirements.txt file.
install_requires = parse_requirements("requirements.txt")


# --- Setup Configuration ---
setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    #long_description=open("README.md","r","UTF-8").read(),
    long_description="",
    long_description_content_type="text/markdown",
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    python_requires=PYTHON_REQUIRES,
    url=URL,
    # Tell setuptools where the package code is
    package_dir={"": "src"},
    packages=packages,
    # Add dependencies from requirements.txt
    install_requires=install_requires,
    # This makes your project's code available to other packages
    include_package_data=True,
    license="MIT", # Or another license of your choice
    # This section creates the command-line script
    entry_points={
        "console_scripts": [
            "video-analyzer=video_analyzer_cli:main",
        ]
    },
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.org/classifiers/
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Video :: Analysis",
    ],
)