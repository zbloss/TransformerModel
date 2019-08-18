from setuptools import setup, find_packages

setup(
    name='TransformerModel',
    version='1.0.0',
    packages=find_packages(exclude=['env']),
    url='https://github.com/zbloss/TransformerModel',
    license='gpl-3.0',
    author='Zachary Bloss',
    author_email='zacharybloss@gmail.com',
    description='A Transformer model implementation in TensorFlow 2.0',
    python_requires='>=3.5',
    install_requires=[
        'matplotlib==3.1.1',
        'numpy',
        'pandas',
        'pyodbc',
        'tensorflow-datasets==1.1.0',
        'tensorflow==2.0.0b1'
    ]
)
