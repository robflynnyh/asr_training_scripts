from setuptools import setup

setup(
    name='speachy',
    version='0.0.0.1',    
    description='Code for training ASR and LMs',
    url='https://github.com/robflynnyh/asr_training_scripts',
    author='Rob Flynn',
    author_email='rjflynn2@sheffield.ac.uk',
    license='BSD 2-clause',
    packages=['speachy'],
    install_requires=['torch',
                      'numpy',                     
                      ],

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',  
        'Operating System :: POSIX :: Linux',        
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
)
