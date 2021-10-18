#!/bin/bash

# Remove existing build
rm -R build/ dist/

# Run new build
python setup.py sdist bdist_wheel