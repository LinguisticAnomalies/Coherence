from setuptools import setup

with open('requirements.txt') as fh:
    requirements = fh.read().splitlines()

setup(
    name='coherencecalculator',
    version='0.1.26pip',
    author='WeizheXu',
    author_email='xuweizhe@uw.edu',
    packages=['coherencecalculator', 'coherencecalculator.pipelines', 'coherencecalculator.tools', 'coherencecalculator.diffcse'],
    package_dir={'coherencecalculator': 'src/coherencecalculator', 
                 'coherencecalculator.pipelines': 'src/coherencecalculator/pipelines', 
                 'coherencecalculator.tools': 'src/coherencecalculator/tools',
                 'coherencecalculator.diffcse': 'src/coherencecalculator/diffcse'},
    package_data={'coherencecalculator': ['src/coherencecalculator/models/*.pickle', 'src/coherencecalculator/vecs/*.bin','src/coherencecalculator/vecs/*.csv']},
    # scripts=['src/scripts'],
    url='http://pypi.python.org/pypi/PackageName/',
    license='LICENSE.txt',
    description='An awesome package that does something',
    long_description=open('README.md').read(),
    install_requires=requirements,
    include_package_data=True
    # dependency_links=['https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.4.0/en_core_web_sm-3.4.0-py3-none-any.whl#egg=en-core-web-sm', 
    #                   'file:///croot/certifi_1671487769961/work/certifi#egg=certifi']
)