from setuptools import setup, find_packages

setup(
    name='pinn_3d_flow_solver',
    version='1.0.0',
    description='Physics-Informed Neural Networks for 3D Navier-Stokes Equations',
    author='',
    author_email='',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    install_requires=[
        'tensorflow==2.20.0',
        'keras==3.12.0',
        'numpy',
        'matplotlib',
        'scipy',
        'h5py',
    ],
    entry_points={
        'console_scripts': [
            'pinn_solver=main:main',
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
