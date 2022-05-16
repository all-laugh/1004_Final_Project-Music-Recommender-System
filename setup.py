#
# Created by Maksim Eremeev (mae9785@nyu.edu)
#

from setuptools import setup, find_packages
import setuptools.command.build_py as build_py


setup_kwargs = dict(
    name='recommender_system',
    version='0.0.1',
    packages=find_packages(),
    install_requires=[],
    setup_requires=[],
    cmdclass={'build_py': build_py.build_py},
)

setup(**setup_kwargs)

