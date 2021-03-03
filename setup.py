# _*_ coding: UTF-8 _*_

import os
from setuptools import setup, find_packages


__all__ = ['setup']


def read_file(filename):
    path = os.path.join(os.path.abspath(os.path.dirname(__file__)), filename)

    try:
        return open(path).readlines()
    except IOError:
        return ''


def get_readme():
    for filename in ('README', 'README.rst', 'README.md'):
        if os.path.exists(filename):
            return read_file(filename).join(os.linesep)

    return ''


setup(
    name='people-tracker-yolo',
    version='0.0.0',
    description='People tracker using Yolo and Deep Sort algorithm.',
    long_description_content_type='text/markdown',
    scripts=[],
    url='https://github.com/edgebr/people-detector-yolo',
    zip_safe=True,
    python_requires='>=3.8',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Operating System :: OS Independent',
        'Development Status :: 4 - Beta',
        'Topic :: Scientific/Engineering',
        'Topic :: System :: Monitoring',
    ],
)