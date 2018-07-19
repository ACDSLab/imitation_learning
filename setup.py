from distutils.core import setup

setup(
    name='imitation_learning',
    version='0.1.0',
    author='K. Lee, K. Saigol, G. Nakajima An, Y. Pan',
    author_email='klee698@gatech.edu, kamilsaigol@gatech.edu, gan9@gatech.edu, ypan37@gatech.edu',
    packages=['imitation_learning'],
    url='https://github.com/ACDSlab/imitation_learning.git',
    license='MIT License',
    description='imlearn: A Python Framework for Imitation Learning',
    long_description=open('README.md').read(),
    install_requires=[
        "numpy",
        "keras",
    ],
)

