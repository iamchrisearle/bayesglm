#### setup.py
from setuptools import setup, find_packages


def get_version():
    with open("your_python_package/__init__.py") as f:
        for line in f:
            if line.startswith("__version__"):
                return line.split("=")[1].strip().strip("'\"")


setup(
    name="bayesglm",
    version="0.1.0",
    author="Chris Earle",
    author_email="chris.c.earle@gmail.com",
    description="A Python translation of the bayesglm model from the R package `arm`",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/iamchrisearle/bayesglm",
    packages=find_packages(),
    install_requires=[
        "statsmodels==0.14.1",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v2 or later (GPLv2+)",
        "Operating System :: OS Independent",
    ],
    license="GPL-2.0-or-later",
    python_requires=">=3.8",
)
