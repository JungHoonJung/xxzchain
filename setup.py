from setuptools import setup, find_packages

setup(
    name             = 'xxzchain',
    version          = '1.0',
    description      = 'Python Quantum spin chain ED library',
    python_requires  = '>=3.6',
    packages         = find_packages(),
    install_requires = ['numpy>=1.16','matplotlib', 'h5py'],


    author           = 'Jung Hoon Jung',
    author_email     = 'jh.jung1380@gmail.com',
    url              = 'https://bitbucket.org/junghoonjung/xxzchain/',
    download_url     = 'https://bitbucket.org/junghoonjung/xxzchain.git'



)
