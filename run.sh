#!/bin/sh
python main.py -c sswe/sfno_sr_dropout_1.ini
python main.py -c sswe/sfno_sr_dropout_2.ini
python main.py -c sswe/sfno_sr_reparam_1.ini
python main.py -c sswe/sfno_sr_reparam_2.ini
