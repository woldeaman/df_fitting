#!/usr/bin/env python
# -*- coding: utf-8 -*-
from setuptools import setup as stsetup


if __name__ == "__main__":
    # poission boltzmann solver
    stsetup(name='DF_fitting',
            packages=['fitting_scripts'],
            version="0.2",
            license='MIT',
            description=('Fitting software for concentration profiles.'),
            author="Amanuel Wolde-Kidan",
            author_email="amanuel.wolde-kidan@fu-berlin.de",
            include_package_data=True,
            zip_safe=False,
            requires=['numpy (>=1.10.4)', 'xlsxwriter (>=1.0.0)', 'matplotlib (>=2.2.2)', 'scipy (>=1.0.1)'],
            install_requires=['numpy>=1.10.4', 'xlsxwriter>=1.0.0', 'matplotlib>=2.2.2', 'scipy>=1.0.1'],
            entry_points={'console_scripts': ['DF_fitting=fitting_scripts.DF_fitting:main', ],},)
