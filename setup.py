from setuptools import setup,find_packages


data_files_to_include = [('', ['README.md', 'LICENSE'])]

setup(name='influ',
      version='0.0.1',
      description='Infer interactions from fluctuations',
      long_description='TODO',
      url='https://github.com/Hallatscheklab/interactions-from-fluctuations',
      author='Takashi Okada, Giulio Isacchini, Oskar Hallatschek',
      author_email='giulioisac@gmail.com',
      license='GPLv3',
      classifiers=[
            'Development Status :: 3 - Alpha',
            'Intended Audience :: Developers',
            'Intended Audience :: Healthcare Industry',
            'Intended Audience :: Science/Research',
            'Topic :: Scientific/Engineering :: Bio-Informatics',
            'Topic :: Scientific/Engineering :: Physics',
            'Topic :: Scientific/Engineering :: Medical Science Apps.',
            'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
            'Natural Language :: English',
            'Programming Language :: Python :: 3.6',
            ],
      packages=find_packages(),
      install_requires=['numpy','cvxpy','scipy','cvxopt'],
#      package_data = {
#            'default_models': [],
#            'default_models/human_T_alpha/': ['sonia/default_models/human_T_alpha/*'],
#            },
      data_files = data_files_to_include,
      include_package_data=True,
#      entry_points = {'console_scripts': [
#            'sonia-evaluate=sonia.evaluate:main'},
      zip_safe=False)
