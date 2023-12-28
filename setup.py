from setuptools import setup

setup(name='diffnn',
      description="Neural Networks using differential geometry metric objects",
      long_description_content_type='text/markdown',
      install_requires=['matplotlib','numpy', 'torch',
                        'progressbar2', 'scipy'
                       ],
      use_scm_version=True,
      setup_requires=['setuptools_scm'],
      packages=['diffnn'],
      include_package_data=True,
      zip_safe=False)

