import setuptools

with open("requirements.txt") as f:
    REQUIRED_PACKAGES = f.read().splitlines()

setuptools.setup(
    name="opt_pmp_utils",
    version="0.0.1",
    author="Felix Frank",
    description="Some utilities for the constrained ProMPs experiments",
    long_description="",
    packages=["opt_pmp_utils"],
    install_requires=REQUIRED_PACKAGES,
    python_requires=">=3.6",
)
