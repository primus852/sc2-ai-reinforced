from setuptools import setup, find_packages


with open('README.rst') as f:
    readme_file = f.read()

with open('LICENSE') as f:
    license_file = f.read()

setup(
    name='sc2-ai-agent',
    version='0.0.2',
    description='A custom Agent for SC2-AI for reinforced learning',
    long_description=readme_file,
    author='Torsten Wolter',
    author_email='tow.berlin@gmail.com',
    url='https://github.com/primus852/sc2-ao-reinforced',
    license=license_file,
    packages=find_packages(exclude=('tests', 'docs'))
)

