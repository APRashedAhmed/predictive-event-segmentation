import versioneer
from setuptools import setup, find_packages

setup(name='predictive-event-segmentation',
      version=versioneer.get_version(),
      cmdclass=versioneer.get_cmdclass(),
      license='Not open source',
      author='apra93',
      packages=find_packages(),
      description='Event segmentation by predictive coding networks',
      )
