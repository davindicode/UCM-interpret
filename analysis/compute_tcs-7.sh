#!/bin/bash

trap "kill 0" EXIT
 
#    'Mouse17': ['130131', '130202'],
#    'Mouse24': ['131218']

# Iterate the string array using for loop
python3 compute_tuning_curves.py --mouse_id Mouse17 --session_id 130131 --gpu 0 &
python3 compute_tuning_curves.py --mouse_id Mouse17 --session_id 130202 --gpu 1 &
python3 compute_tuning_curves.py --mouse_id Mouse24 --session_id 131218 --gpu 1
