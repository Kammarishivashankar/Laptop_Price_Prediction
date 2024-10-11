from setuptools import find_packages,setup
from typing import List

HYPEN_E_DOT = '-e.'
def install_requirements(file_path)->List[str]:
    """reads the required libraries and returns them as a list of strings"""
    requirements = []
    with open(file_path,'r') as file:
        requirements = file.readlines()
        [libraries.replace("\n","") for libraries in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
    return requirements




setup(
    name = 'ml_regression_preoject',
    version = '0.0.1',
    author = 'Kammari Shiva Shankar',
    author_email='kammarishivashankarr@gmail.com',
    packages=find_packages(),
    install_requires = install_requirements('requirements.txt')
)