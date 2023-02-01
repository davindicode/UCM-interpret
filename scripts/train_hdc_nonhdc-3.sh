#!/bin/bash

trap "kill 0" EXIT


# Declare an array of string with type
#    'Mouse28': ['140310', '140311', '140312', '140313', '140317', '140318'],
declare -a sessions1=("140310" "140311")
 
# Iterate the string array using for loop
for val in ${sessions1[@]}
do
    python3 HDC.py --checkpoint_dir /scratches/ramanujan_2/vn283/HDC_PartIII/checkpoint/ --data_path /scratches/ramanujan_2/dl543/HDC_PartIII/ --mouse_id Mouse28 --session_id $val --phase wake --subset hdc --cv_folds 5 --cv -1 --ncvx 2 --batch_size 5000 --max_epochs 3000 --bin_size 160 --likelihood U-ec-3 --mapping svgp-64 --x_mode hd-omega-speed-x-y-time --lr 1e-2 --jitter 1e-4 --gpu 0 &
    python3 HDC.py --checkpoint_dir /scratches/ramanujan_2/vn283/HDC_PartIII/checkpoint/ --data_path /scratches/ramanujan_2/dl543/HDC_PartIII/ --mouse_id Mouse28 --session_id $val --phase wake --subset nonhdc --cv_folds 5 --cv -1 --ncvx 2 --batch_size 500 --max_epochs 3000 --bin_size 160 --likelihood U-ec-3 --mapping svgp-64 --x_mode hd-omega-speed-x-y-time --lr 1e-2 --jitter 1e-4 --gpu 1;
done

#declare -a sessions2=("130514" "130515")
#for val in ${sessions2[@]}
#do
#    python3 HDC.py --checkpoint_dir /scratches/ramanujan_2/vn283/HDC_PartIII/checkpoint/ --data_path /scratches/ramanujan_2/dl543/HDC_PartIII/ --mouse_id Mouse20 --session_id $val --phase wake --subset nonhdc --cv_folds 5 --cv -1 --ncvx 2 --batch_size 1000 --max_epochs 3000 --bin_size 160 --likelihood U-ec-3 --mapping svgp-64 --x_mode hd-omega-speed-x-y-time --lr 1e-2 --jitter 1e-4 --gpu 0 ;
#    python3 HDC.py --checkpoint_dir /scratches/ramanujan_2/vn283/HDC_PartIII/checkpoint/ --data_path /scratches/ramanujan_2/dl543/HDC_PartIII/ --mouse_id Mouse17 --session_id $val --phase wake --subset hdc --cv_folds 5 --cv -1 --ncvx 2 --batch_size 5000 --max_epochs 3000 --bin_size 160 --likelihood U-ec-3 --mapping svgp-64 --x_mode hd-omega-speed-x-y-time --lr 1e-2 --jitter 1e-4 --gpu 1;
#done
