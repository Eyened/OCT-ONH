from setuptools import find_packages, setup

with open('requirements.txt') as f:
    install_requires = f.read().splitlines()

setup(
    name="OCT-ONH",
    version="1.0",
    author="Karin van Garderen",
    author_email="k.vangarderen@erasmusmc.nl",
    description="A package to extract optic nerve head features from OCT volumes.",
    packages=find_packages(),
    install_requires=install_requires,
    package_data={},
)
