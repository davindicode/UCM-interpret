#!/bin/bash
 
# Declare an array of string with type
declare -a sessions=("131216" "131217" "131218")
 
# Iterate the string array using for loop
for val in ${sessions[@]}; do
    python3 HDC.py --checkpoint_dir /scratches/ramanujan_2/vn283/HDC_PartIII/checkpoint/ --data_path /scratches/ramanujan_2/dl543/HDC_PartIII/ --mouse_id Mouse24 --session_id $val --phase wake --subset hdc --cv_folds 5 --cv -1 --ncvx 2 --batch_size 500 --max_epochs 2 --bin_size 160 --likelihood U-ec-3 --mapping svgp-64 --x_mode hd-omega-speed-x-y-time --lr 1e-2 --jitter 1e-5 --gpu 0 
done