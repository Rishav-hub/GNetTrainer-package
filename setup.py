from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
  name = 'GNetTrainer',
  packages = find_packages(),
  include_package_data=True,
  version = '2.7',
  license='MIT',
  description = 'Image classification trainer',
  author = 'Rishav-hub',
  author_email = '9930046@gmail.com',
 url = 'https://github.com/Rishav-hub/GNetTrainer-package.git',
 # download_url = 'https://github.com/sourangshupal/pyarithcalc/releases/download/0.5/pyarithcalc-0.5.tar.gz',
  keywords = ['gnetrainer'],
  install_requires=[
        'tensorflow==2.4.1',
        'scipy==1.6.3',
        'numpy',
        'pandas',
        'Pillow==8.3.2',
        'Flask',
        'Flask-Cors==3.0.10'
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',

    # Indicate who your project is intended for
    'Intended Audience :: Developers',
    'Topic :: Software Development :: Build Tools',

    # Pick your license as you wish
    'License :: OSI Approved :: MIT License',

    # Specify the Python versions you support here. In particular, ensure
    # that you indicate whether you support Python 2, Python 3 or both.
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
  ],
  entry_points={
        "console_scripts": [
            "gnet = GNetTrainer.main:start_app",
        ]},
)
