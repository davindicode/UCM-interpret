#!/bin/bash

trap "kill 0" EXIT
 
#'Mouse17': ['130125', '130128', '130131', '130202', '130203']
#'Mouse20': ['130514', '130515', '130516', '130517'],
#'Mouse24': ['131213', '131217', '131218'],
#'Mouse25': ['140124', '140128', '140129'],


declare -a sessions1=("131213" "131217" "131218")

python3 compute_SCDs.py --mouse_id Mouse12 --session_id 120807 --gpu 0 ;

# Iterate the string array using for loop
#for val in ${sessions1[@]}
#do
#    python3 compute_SCDs.py --mouse_id Mouse24 --session_id $val --gpu 0 ;
#done

#declare -a sessions2=("140124" "140128" "140129")

# Iterate the string array using for loop
#for val in ${sessions2[@]}
#do
#    python3 compute_SCDs.py --mouse_id Mouse25 --session_id $val --gpu 0 ;
#done


#python3 compute_SCDs.py --mouse_id Mouse28 --session_id 140310 --gpu 0 ;

