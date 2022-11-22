from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="aqmlator",
    version="0.0.0",
    author="Tomasz Rybotycki",
    author_email="rybotycki.tomasz+aqmlator@gmail.com",
    long_description=long_description,
    long_description_content_type="text/markdown",
    description="A package for auto quantum machine learning-izing your experiments!",
    packages=find_packages(exclude=["tests"]),
    install_requires=[
        "requests",
    ],
    classifiers=[
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Natural Language :: English",
    ],
    license="Apache License 2.0.",
)
