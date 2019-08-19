from setuptools import setup, find_packages

setup(
    name='transformer_model',
    version='1.2.0',
    packages=find_packages(exclude=['env']),
    url='https://github.com/zbloss/TransformerModel',
    download_url='https://github.com/zbloss/TransformerModel/archive/1.1.0.tar.gz',
    keywords=['Transformer', 'seq2seq', 'SelfAttention'],
    license='mit',
    author='Zachary Bloss',
    author_email='zacharybloss@gmail.com',
    description='A Transformer model implementation in TensorFlow 2.0',
    python_requires='>=3.5',
    install_requires=[
        'matplotlib',
        'numpy',
        'pandas',
        'pyodbc',
        'tensorflow-datasets>=1.1.0',
        'tensorflow>=2.0.0b1'
    ],
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
      ],
)
