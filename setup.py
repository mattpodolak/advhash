import setuptools
import re

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

VERSION_FILE = "advhash/__init__.py"
with open(VERSION_FILE) as version_file:
    match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                      version_file.read(), re.MULTILINE)

if match:
    version = match.group(1)
else:
    raise RuntimeError(f"Unable to find version string in {VERSION_FILE}.")

setuptools.setup(
    name="advhash",
    version=version,
    author="Matthew Podolak",
    author_email="mpodola2@gmail.com",
    description="Adversarial attacks for perceptual image hashing functions",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mattpodolak/advhash",
    packages=setuptools.find_packages(),
    license='GNU GPLv3',
    install_requires=['requests'],
    keywords='adversarial attacks image hashing',
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3 :: Only",
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.5',
)