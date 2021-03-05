import setuptools

DESCRIPTION = ""
DISTNAME = 'datatlon'
MAINTAINER = 'Martynas Miliauskas'
MAINTAINER_EMAIL = 'mmiliauskas@protonmail.com'
VERSION = '1.0.0'
LICENSE = ""
URL = ""
DOWNLOAD_URL=""
INSTALL_REQUIRES = [
        'numpy>=1.19.3',
        'pandas>=1.1.5',
        'tabulate>=0.8.7',
        'arrow>=0.15.8'
]
PYTHON_REQUIRES='>=3.6'
CLASSIFIERS =[
    "Programming Language :: Python :: 3",
    "Operating System :: OS Independent",
]

try:
    from setuptools import setup
    _has_setuptools = True
except ImportError:
    from distutils.core import setup

if __name__ == "__main__":
    setup(
        name=DISTNAME,
        author=MAINTAINER,
        author_email=MAINTAINER_EMAIL,
        maintainer=MAINTAINER,
        maintainer_email=MAINTAINER_EMAIL,
        description=DESCRIPTION,
        long_description=DESCRIPTION,
        license=LICENSE,
        url=URL,
        version=VERSION,
        download_url=DOWNLOAD_URL,
        install_requires=INSTALL_REQUIRES,
        packages=setuptools.find_packages(),
        classifiers=CLASSIFIERS,
        python_requires=PYTHON_REQUIRES
    )