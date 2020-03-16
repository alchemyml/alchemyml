import setuptools

with open('README.md', 'r') as file:
    long_description = file.read()

setuptools.setup(
    name='alchemyml',  
    version='0.1',
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
) 
    
