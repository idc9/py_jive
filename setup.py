from setuptools import setup, find_packages


def readme():
    with open('README.rst') as f:
        return f.read()


install_requires = ['numpy', 'scipy', 'pandas', 'seaborn', 'matplotlib', 'sklearn', 'py_fun_iain']

setup(name='jive',
      version='0.0.1',
      description='Implementation of the Angle based Joint and Inidividual Variation Explained',
      url='http://github.com/idc9/jive',
      author='Iain Carmichael',
      author_email='idc9@cornell.edu',
      license='MIT',
      packages=find_packages(),
      install_requires=install_requires,
      test_suite='nose.collector',
      tests_require=['nose'],
      zip_safe=False)
