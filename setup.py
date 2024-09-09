from setuptools import find_packages, setup

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = fh.read()

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="aqmlator",
    version="0.1.2",
    author="Tomasz Rybotycki",
    author_email="rybotycki.tomasz+aqmlator@gmail.com",
    long_description=long_description,
    long_description_content_type="text/markdown",
    description="A package for auto quantum machine learning-izing your experiments!",
    packages=find_packages(exclude=["tests"]),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    license="Apache License 2.0.",
)
