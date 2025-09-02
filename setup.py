"""
Setup configuration for PitchPerfect AI package
"""

from setuptools import setup, find_packages
import os

# Read the contents of README file
this_directory = os.path.abspath(os.path.dirname(__file__))
try:
    with open(os.path.join(this_directory, "README.md"), encoding="utf-8") as f:
        long_description = f.read()
except FileNotFoundError:
    long_description = "AI-powered speech analysis and improvement system"


# Read requirements from requirements.txt
def read_requirements():
    with open("requirements.txt", "r") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]


setup(
    name="pitchperfect",
    version="0.1.0",
    author="PitchPerfect Team",
    author_email="team@pitchperfect.ai",
    description="AI-powered speech analysis and improvement system",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tanzania2025/pitch_perfect",
    packages=find_packages(where="."),
    package_dir={"": "."},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Sound/Audio :: Speech",
    ],
    python_requires=">=3.8,<3.13",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "isort>=5.12.0",
            "pre-commit>=3.3.0",
        ],
        "docs": [
            "sphinx>=6.0.0",
            "sphinx-rtd-theme>=1.3.0",
            "sphinxcontrib-napoleon>=0.7",
        ],
    },
    entry_points={
        "console_scripts": [
            "pitchperfect=pitchperfect.cli:main",
            "pitchperfect-demo=pitchperfect.scripts.demo:main",
            "pitchperfect-preprocess=pitchperfect.scripts.preprocess_meld:main",
        ],
    },
    include_package_data=True,
    package_data={
        "pitchperfect": [
            "config/*.yaml",
            "config/*.json",
            "templates/*.txt",
        ],
    },
    zip_safe=False,
    keywords="speech-analysis, sentiment-analysis, text-to-speech, nlp, ai",
    project_urls={
        "Bug Reports": "https://github.com/tanzania2025/pitch_perfect/issues",
        "Source": "https://github.com/tanzania2025/pitch_perfect",
        "Documentation": "https://pitch-perfect.readthedocs.io/",
    },
)
