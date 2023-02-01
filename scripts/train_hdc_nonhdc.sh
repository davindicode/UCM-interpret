#!/bin/bash

trap "kill 0" EXIT


# Declare an array of string with type
#    'Mouse12': ['120809', '120810'],
declare -a sessions1=("120810")

python3 HDC.py --checkpoint_dir /scratches/ramanujan_2/vn283/HDC_PartIII/checkpoint/ --data_path /scratches/ramanujan_2/dl543/HDC_PartIII/ --mouse_id Mouse12 --session_id 120810 --phase wake --subset nonhdc --cv_folds 5 --cv -1 --ncvx 2 --batch_size 500 --max_epochs 3000 --bin_size 160 --likelihood U-ec-3 --mapping svgp-64 --x_mode hd-omega-speed-x-y-time --lr 1e-2 --jitter 1e-4 --gpu 0 &

python3 HDC.py --checkpoint_dir /scratches/ramanujan_2/vn283/HDC_PartIII/checkpoint/ --data_path /scratches/ramanujan_2/dl543/HDC_PartIII/ --mouse_id Mouse12 --session_id 120810 --phase wake --subset hdc --cv_folds 5 --cv -1 --ncvx 2 --batch_size 1000 --max_epochs 3000 --bin_size 160 --likelihood U-ec-3 --mapping svgp-64 --x_mode hd-omega-speed-x-y-time --lr 1e-2 --jitter 1e-4 --gpu 0 

# Iterate the string array using for loop
#for val in ${sessions1[@]}
#do
    #python3 HDC.py --checkpoint_dir /scratches/ramanujan_2/vn283/HDC_PartIII/checkpoint/ --data_path /scratches/ramanujan_2/dl543/HDC_PartIII/ --mouse_id Mouse12 --session_id $val --phase wake --subset hdc --cv_folds 5 --cv -1 --ncvx 2 --batch_size 2000 --max_epochs 3000 --bin_size 160 --likelihood U-ec-3 --mapping svgp-64 --x_mode hd-omega-speed-x-y-time --lr 1e-2 --jitter 1e-4 --gpu 0 ;
#    python3 HDC.py --checkpoint_dir /scratches/ramanujan_2/vn283/HDC_PartIII/checkpoint/ --data_path /scratches/ramanujan_2/dl543/HDC_PartIII/ --mouse_id Mouse12 --session_id $val --phase wake --subset nonhdc --cv_folds 5 --cv -1 --ncvx 2 --batch_size 100 --max_epochs 3000 --bin_size 160 --likelihood U-ec-3 --mapping svgp-64 --x_mode hd-omega-speed-x-y-time --lr 1e-2 --jitter 1e-4 --gpu 1;
#done
