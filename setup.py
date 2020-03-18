import setuptools

import re
VERSIONFILE = "alchemyml/_version.py"
verstrline = open(VERSIONFILE, "rt").read()
VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
mo = re.search(VSRE, verstrline, re.M)
if mo:
    verstr = mo.group(1)
else:
    raise RuntimeError("Unable to find version string in %s." % (VERSIONFILE,))

with open('README.md', 'r') as file:
    long_description = file.read()

setuptools.setup(
    name='alchemyml',  
    version=verstr,
    author="Alchemy Machine Learning, S. L.",
    author_email="admin@alchemyml.com",
    description="AlchemyML API package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/alchemyml/alchemyml",
    packages=setuptools.find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
    include_package_data=True,
) 
    
