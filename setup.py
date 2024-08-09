from setuptools import find_packages, setup
from typing import List


HYPEN_DOT ="-e."

def get_requirements(file_path : str) -> List[str]:
    """this function will return the list of requirements
    """
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [requirement.strip().replace("\n","") for requirement in requirements]
    if HYPEN_DOT in requirements:
        requirements.remove(HYPEN_DOT)
    return requirements

setup (
    name = "Querymate",
    version = "0.1.0",
    author = "Akshay Soner",
    author_email = "akshay.soner@thecodewise.com",
    packages = find_packages(),
    install_requires = get_requirements('requirements.txt'))
    