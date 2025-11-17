"""
Setup script for the Audio Language Model (ALM) project
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = fh.read().splitlines()

setup(
    name="audio-language-model",
    version="1.0.0",
    author="DRDO, MoD",
    author_email="contact@drdo.gov.in",
    description="A deep learning based Audio Language Model for joint speech and non-speech audio understanding",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/drdo-mod/alm",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Sound/Audio :: Speech",
        "Topic :: Multimedia :: Sound/Audio :: Analysis",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "alm-train=training.train:main",
            "alm-infer=main:main",
            "alm-generate-dataset=utils.dataset_generator:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)