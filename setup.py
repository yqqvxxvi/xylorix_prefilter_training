"""Setup script for wood classification package"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / 'README.md'
long_description = readme_file.read_text() if readme_file.exists() else ''

setup(
    name='wood-classification',
    version='1.0.0',
    description='Wood microscopy classification using ML and deep learning',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Your Name',
    author_email='your.email@example.com',
    url='https://github.com/yourusername/wood-classification',
    packages=find_packages(),
    python_requires='>=3.8',
    install_requires=[
        'torch>=2.0.0',
        'torchvision>=0.15.0',
        'numpy>=1.24.0',
        'pandas>=2.0.0',
        'opencv-python>=4.8.0',
        'scikit-learn>=1.3.0',
        'scipy>=1.11.0',
        'scikit-image>=0.21.0',
        'tqdm>=4.66.0',
        'pyyaml>=6.0',
        'joblib>=1.3.0',
    ],
    extras_require={
        'dev': [
            'matplotlib>=3.7.0',
            'seaborn>=0.12.0',
            'jupyter>=1.0.0',
            'pytest>=7.4.0',
        ]
    },
    entry_points={
        'console_scripts': [
            'wood-train-ml=scripts.train_blob_ml:main',
            'wood-train-cnn=scripts.train_cnn:main',
            'wood-predict=scripts.predict:main',
        ],
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
)
