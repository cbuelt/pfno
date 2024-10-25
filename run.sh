#!/bin/sh
python main.py -c darcy_flow/fno_timing.ini
python main.py -c darcy_flow/uno_timing.ini
python main.py -c ks/fno_timing.ini
python main.py -c ks/uno_timing.ini
python main.py -c era5/fno_timing.ini
python main.py -c sswe/sfno_timing.ini

