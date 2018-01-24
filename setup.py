from setuptools import setup, find_packages


def readme():
    with open('README.rst') as f:
        return f.read()


install_requires = ['numpy', 'scipy', 'pandas', 'seaborn', 'matplotlib', 'sklearn']

setup(name='jive',
      version='0.0.2',
      description='Implementation of the Angle based Joint and Inidividual Variation Explained',
      url='https://github.com/idc9/py_jive',
      download_url = 'https://github.com/idc9/py_jive/tarball/0.0.2',
      author='Iain Carmichael',
      author_email='idc9@cornell.edu',
      license='MIT',
      packages=find_packages(),
      install_requires=install_requires,
      test_suite='nose.collector',
      tests_require=['nose'],
      zip_safe=False)
