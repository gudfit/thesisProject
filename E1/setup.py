"""Setup script for LLM Compression project."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [
        line.strip() for line in fh if line.strip() and not line.startswith("#")
    ]

setup(
    name="llm-compression",
    version="1.0.0",
    author="Gudfit",
    author_email="daniel2468114@gmail.com",
    description="Large Language Models as Data Compressors - MSc Project",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/gudfit/llm-compression",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "llm-compress=experiments.run_experiments:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["config/*.yaml"],
    },
)
