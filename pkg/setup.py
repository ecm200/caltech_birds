from setuptools import setup, find_packages

setup(name='cub_tools',
      version='2.5.9',
      description='Caltech UCSD Birds Database Tools for PyTorch Image Classification',
      url='https://github.com/ecm200/caltech_birds',
      author='Ed Morris',
      author_email='ecm200@gmail.com',
      license='MIT',
      #package_dir={'': 'cub_tools'},
      packages=find_packages(),
      #packages=['cub_tools'],
      zip_safe=False,
      install_requires=[
            #'pytorch', # was set to 1.4, try latest
            #'torchvision',
            'imutils',
            'pandas',
            'matplotlib',
            'numpy',
            'torch-lucent',
            'pytorchcv',
            'scikit-image',
            'Pillow',
            'yacs',
            'torchsummary',
            'ignite'],
      python_requires=">=3.7"
      )
