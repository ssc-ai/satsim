#!/bin/bash
python3 -m pip install -U -r requirements_dev.txt
pip3 install .
python3 setup.py develop
source ~/.profile
make docs
make dist
