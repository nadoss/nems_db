import codecs
import os.path
from setuptools import find_packages, setup

NAME = 'nems_db'

VERSION = 'pre-alpha'

with codecs.open('README.md', encoding='utf-8') as f:
    long_description = f.read()

GENERAL_REQUIRES = ['flask', 'flask_restful', 'numpy',
                    'scipy', 'matplotlib', 'pandas','sqlalchemy','pymysql']

setup(
    name=NAME,
    version=VERSION,
    packages=find_packages(exclude=['tests']),
    include_package_data=True,
    zip_safe=True,
    author='LBHB',
    author_email='lbhb.ohsu@gmail.com',
    description='Neural Encoding Model System - Database support',
    long_description=long_description,
    url='http://neuralprediction.org',
    install_requires=GENERAL_REQUIRES,
    classifiers=[],
    entry_points={
        'console_scripts': [
            #'nems-fit-single=nems.application.cmd_launcher:main',
        ],
    }
)
