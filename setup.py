from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

# Get the long description from the README file
long_description = (here / "README.md").read_text(encoding="utf-8")

setup(
    name="duckietownrl",
    version="1.0.0",
    description="A python library for RL with DuckieTown",
    url="https://github.com/gferraro2019/duckietown_rl_course",
    author="Giuseppe Ferraro",
    author_email="giuseppe.ferraro@isae-supaero.fr",
    packages=find_packages(),
    python_requires=">=3.8, <3.9",
)
