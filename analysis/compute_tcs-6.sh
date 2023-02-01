#!/bin/bash

trap "kill 0" EXIT
 
#     'Mouse20': ['130515', '130516', '130517'],

# Iterate the string array using for loop
#python3 compute_tuning_curves.py --mouse_id Mouse28 --session_id 140310 --gpu 0
python3 compute_tuning_curves.py --mouse_id Mouse17 --session_id 130203 --gpu 1