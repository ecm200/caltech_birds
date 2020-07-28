from setuptools import setup

setup(name='cub_tools',
      version='0.1',
      description='Caltech UCSD Birds Database Tools for PyTorch Image Classification',
      url='https://github.com/ecm200/caltech_birds',
      author='Ed Morris',
      author_email='ecm200@gmail.com',
      license='MIT',
      packages=['cub_tools'],
      install_requires=[
            'pytorch=1.4',
            'torchvision',
            'imutils',
            'pandas',
            'matplotlib',
            'numpy',
            'torch-lucent',
            'pytorchcv',
            'scikit-image',
            'Pillow']
      zip_safe=False)
