from setuptools import setup
from Cython.Build import cythonize


setup(
    name="decontaminator",
    ext_modules=cythonize("decontaminator/cython/*.pyx", language_level="3", language="c++"),
)
