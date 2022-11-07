#!/bin/bash



echo "Running block of 5 models..."

python3 HDC.py --checkpoint_dir /scratches/ramanujan_2/dl543/HDC_PartIII/checkpoint/ --data_path /scratches/ramanujan_2/dl543/HDC_PartIII/ --mouse_id Mouse12 --session_id 120806 --phase wake --subset hdc --cv_folds 5 --cv 0 1 2 3 4 -1 --ncvx 2 --batch_size 10000 --max_epochs 2 --bin_size 160 --likelihood U-ec-3 --mapping svgp-64 --x_mode hd-omega-speed-x-y-time --lr 1e-2 --jitter 1e-5 --gpu 0 &
BACK_PID1=$!

python3 HDC.py --session_id 12-120806 --phase wake --cv_folds 5 --cv 0 1 2 3 4 -1 --ncvx 2 --batch_size 9000 --bin_size 30 --likelihood Uqd3 --mapping svgp64 --x_mode hd-w-s-pos-t --lr 1e-2 --jitter 1e-5 --gpu 0 &
BACK_PID2=$!

python3 HDC.py --session_id 12-120806 --phase wake --cv_folds 5 --cv 0 1 2 3 4 -1 --ncvx 2 --batch_size 5000 --bin_size 50 --likelihood Uqd3 --mapping svgp64 --x_mode hd-w-s-pos-t --lr 1e-2 --jitter 1e-5 --gpu 1 &
BACK_PID3=$!

python3 HDC.py --session_id 12-120806 --phase wake --cv_folds 5 --cv 0 1 2 3 4 -1 --ncvx 2 --batch_size 4000 --bin_size 70 --likelihood Uqd3 --mapping svgp64 --x_mode hd-w-s-pos-t --lr 1e-2 --jitter 1e-5 --gpu 1 &
BACK_PID4=$!

python3 HDC.py --session_id 12-120806 --phase wake --cv_folds 5 --cv 0 1 2 3 4 -1 --ncvx 2 --batch_size 3000 --bin_size 90 --likelihood Uqd3 --mapping svgp64 --x_mode hd-w-s-pos-t --lr 1e-2 --jitter 1e-5 --gpu 1 &
BACK_PID5=$!

wait $BACK_PID1
wait $BACK_PID2
wait $BACK_PID3
wait $BACK_PID4
wait $BACK_PID5

echo "Block done."





 'Mouse24': ['131213', '131216', '131217','131218'],
    'Mouse25': ['140123', '140124', '140128', '140129', #'140130', '140131', '140203', '140204', '140205', '140206'],
    'Mouse28': ['140310', '140311', '140312', '140313', #'140317', '140318'],
    
    

python3 HDC.py --checkpoint_dir /scratches/ramanujan_2/dl543/HDC_PartIII/checkpoint/ --data_path /scratches/ramanujan_2/dl543/HDC_PartIII/ --mouse_id Mouse24 --session_id 131213 --phase wake --subset hdc --cv_folds 5 --cv -1 --ncvx 3 --batch_size 5000 --max_epochs 3000 --bin_size 160 --likelihood U-ec-3 --mapping svgp-64 --x_mode hd-omega-speed-x-y-time --lr 1e-2 --jitter 1e-5 --gpu 0 &

python3 HDC.py --checkpoint_dir /scratches/ramanujan_2/dl543/HDC_PartIII/checkpoint/ --data_path /scratches/ramanujan_2/dl543/HDC_PartIII/ --mouse_id Mouse24 --session_id 131213 --phase wake --subset nonhdc --cv_folds 5 --cv -1 --ncvx 3 --batch_size 2000 --max_epochs 3000 --bin_size 160 --likelihood U-ec-3 --mapping svgp-64 --x_mode hd-omega-speed-x-y-time --lr 1e-2 --jitter 1e-5 --gpu 0 &

python3 HDC.py --checkpoint_dir /scratches/ramanujan_2/dl543/HDC_PartIII/checkpoint/ --data_path /scratches/ramanujan_2/dl543/HDC_PartIII/ --mouse_id Mouse24 --session_id 131216 --phase wake --subset hdc --cv_folds 5 --cv -1 --ncvx 3 --batch_size 5000 --max_epochs 3000 --bin_size 160 --likelihood U-ec-3 --mapping svgp-64 --x_mode hd-omega-speed-x-y-time --lr 1e-2 --jitter 1e-5 --gpu 1 &

python3 HDC.py --checkpoint_dir /scratches/ramanujan_2/dl543/HDC_PartIII/checkpoint/ --data_path /scratches/ramanujan_2/dl543/HDC_PartIII/ --mouse_id Mouse24 --session_id 131216 --phase wake --subset nonhdc --cv_folds 5 --cv -1 --ncvx 3 --batch_size 1000 --max_epochs 3000 --bin_size 160 --likelihood U-ec-3 --mapping svgp-64 --x_mode hd-omega-speed-x-y-time --lr 1e-2 --jitter 1e-5 --gpu 1 &






python3 HDC.py --checkpoint_dir /scratches/ramanujan_2/dl543/HDC_PartIII/checkpoint/ --data_path /scratches/ramanujan_2/dl543/HDC_PartIII/ --mouse_id Mouse25 --session_id 140123 --phase wake --subset hdc --cv_folds 5 --cv -1 --ncvx 3 --batch_size 5000 --max_epochs 3000 --bin_size 160 --likelihood U-ec-3 --mapping svgp-64 --x_mode hd-omega-speed-x-y-time --lr 1e-2 --jitter 1e-5 --gpu 0 &

python3 HDC.py --checkpoint_dir /scratches/ramanujan_2/dl543/HDC_PartIII/checkpoint/ --data_path /scratches/ramanujan_2/dl543/HDC_PartIII/ --mouse_id Mouse25 --session_id 140123 --phase wake --subset nonhdc --cv_folds 5 --cv -1 --ncvx 3 --batch_size 1000 --max_epochs 3000 --bin_size 160 --likelihood U-ec-3 --mapping svgp-64 --x_mode hd-omega-speed-x-y-time --lr 1e-2 --jitter 1e-5 --gpu 0 &

python3 HDC.py --checkpoint_dir /scratches/ramanujan_2/dl543/HDC_PartIII/checkpoint/ --data_path /scratches/ramanujan_2/dl543/HDC_PartIII/ --mouse_id Mouse25 --session_id 140124 --phase wake --subset hdc --cv_folds 5 --cv -1 --ncvx 3 --batch_size 5000 --max_epochs 3000 --bin_size 160 --likelihood U-ec-3 --mapping svgp-64 --x_mode hd-omega-speed-x-y-time --lr 1e-2 --jitter 1e-5 --gpu 1 &

python3 HDC.py --checkpoint_dir /scratches/ramanujan_2/dl543/HDC_PartIII/checkpoint/ --data_path /scratches/ramanujan_2/dl543/HDC_PartIII/ --mouse_id Mouse25 --session_id 140124 --phase wake --subset nonhdc --cv_folds 5 --cv -1 --ncvx 3 --batch_size 1000 --max_epochs 3000 --bin_size 160 --likelihood U-ec-3 --mapping svgp-64 --x_mode hd-omega-speed-x-y-time --lr 1e-2 --jitter 1e-5 --gpu 1 &




python3 HDC.py --checkpoint_dir /scratches/ramanujan_2/dl543/HDC_PartIII/checkpoint/ --data_path /scratches/ramanujan_2/dl543/HDC_PartIII/ --mouse_id Mouse28 --session_id 140310 --phase wake --subset hdc --cv_folds 5 --cv -1 --ncvx 3 --batch_size 5000 --max_epochs 3000 --bin_size 160 --likelihood U-ec-3 --mapping svgp-64 --x_mode hd-omega-speed-x-y-time --lr 1e-2 --jitter 1e-5 --gpu 0 &

python3 HDC.py --checkpoint_dir /scratches/ramanujan_2/dl543/HDC_PartIII/checkpoint/ --data_path /scratches/ramanujan_2/dl543/HDC_PartIII/ --mouse_id Mouse28 --session_id 140310 --phase wake --subset nonhdc --cv_folds 5 --cv -1 --ncvx 3 --batch_size 1000 --max_epochs 3000 --bin_size 160 --likelihood U-ec-3 --mapping svgp-64 --x_mode hd-omega-speed-x-y-time --lr 1e-2 --jitter 1e-5 --gpu 0 &

python3 HDC.py --checkpoint_dir /scratches/ramanujan_2/dl543/HDC_PartIII/checkpoint/ --data_path /scratches/ramanujan_2/dl543/HDC_PartIII/ --mouse_id Mouse28 --session_id 140311 --phase wake --subset hdc --cv_folds 5 --cv -1 --ncvx 3 --batch_size 3000 --max_epochs 3000 --bin_size 160 --likelihood U-ec-3 --mapping svgp-64 --x_mode hd-omega-speed-x-y-time --lr 1e-2 --jitter 1e-5 --gpu 1 &

python3 HDC.py --checkpoint_dir /scratches/ramanujan_2/dl543/HDC_PartIII/checkpoint/ --data_path /scratches/ramanujan_2/dl543/HDC_PartIII/ --mouse_id Mouse28 --session_id 140311 --phase wake --subset nonhdc --cv_folds 5 --cv -1 --ncvx 3 --batch_size 500 --max_epochs 3000 --bin_size 160 --likelihood U-ec-3 --mapping svgp-64 --x_mode hd-omega-speed-x-y-time --lr 1e-2 --jitter 1e-5 --gpu 1 &





python3 HDC.py --checkpoint_dir /scratches/ramanujan_2/dl543/HDC_PartIII/checkpoint/ --data_path /scratches/ramanujan_2/dl543/HDC_PartIII/ --mouse_id Mouse28 --session_id 140312 --phase wake --subset hdc --cv_folds 5 --cv -1 --ncvx 3 --batch_size 3000 --max_epochs 3000 --bin_size 160 --likelihood U-ec-3 --mapping svgp-64 --x_mode hd-omega-speed-x-y-time --lr 1e-2 --jitter 1e-5 --gpu 0 &

python3 HDC.py --checkpoint_dir /scratches/ramanujan_2/dl543/HDC_PartIII/checkpoint/ --data_path /scratches/ramanujan_2/dl543/HDC_PartIII/ --mouse_id Mouse28 --session_id 140313 --phase wake --subset hdc --cv_folds 5 --cv -1 --ncvx 3 --batch_size 2000 --max_epochs 3000 --bin_size 160 --likelihood U-ec-3 --mapping svgp-64 --x_mode hd-omega-speed-x-y-time --lr 1e-2 --jitter 1e-5 --gpu 0 &

python3 HDC.py --checkpoint_dir /scratches/ramanujan_2/dl543/HDC_PartIII/checkpoint/ --data_path /scratches/ramanujan_2/dl543/HDC_PartIII/ --mouse_id Mouse25 --session_id 140128 --phase wake --subset hdc --cv_folds 5 --cv -1 --ncvx 3 --batch_size 10000 --max_epochs 3000 --bin_size 160 --likelihood U-ec-3 --mapping svgp-64 --x_mode hd-omega-speed-x-y-time --lr 1e-2 --jitter 1e-5 --gpu 1 &

python3 HDC.py --checkpoint_dir /scratches/ramanujan_2/dl543/HDC_PartIII/checkpoint/ --data_path /scratches/ramanujan_2/dl543/HDC_PartIII/ --mouse_id Mouse25 --session_id 140128 --phase wake --subset nonhdc --cv_folds 5 --cv -1 --ncvx 3 --batch_size 1000 --max_epochs 3000 --bin_size 160 --likelihood U-ec-3 --mapping svgp-64 --x_mode hd-omega-speed-x-y-time --lr 1e-2 --jitter 1e-5 --gpu 1 &







python3 HDC.py --checkpoint_dir /scratches/ramanujan_2/dl543/HDC_PartIII/checkpoint/ --data_path /scratches/ramanujan_2/dl543/HDC_PartIII/ --mouse_id Mouse28 --session_id 140312 --phase wake --subset nonhdc --cv_folds 5 --cv -1 --ncvx 3 --batch_size 1000 --max_epochs 3000 --bin_size 160 --likelihood U-ec-3 --mapping svgp-64 --x_mode hd-omega-speed-x-y-time --lr 1e-2 --jitter 1e-5 --gpu 0 &

python3 HDC.py --checkpoint_dir /scratches/ramanujan_2/dl543/HDC_PartIII/checkpoint/ --data_path /scratches/ramanujan_2/dl543/HDC_PartIII/ --mouse_id Mouse28 --session_id 140313 --phase wake --subset nonhdc --cv_folds 5 --cv -1 --ncvx 3 --batch_size 1000 --max_epochs 3000 --bin_size 160 --likelihood U-ec-3 --mapping svgp-64 --x_mode hd-omega-speed-x-y-time --lr 1e-2 --jitter 1e-5 --gpu 1 &





python3 HDC.py --checkpoint_dir /scratches/ramanujan_2/dl543/HDC_PartIII/checkpoint/ --data_path /scratches/ramanujan_2/dl543/HDC_PartIII/ --mouse_id Mouse25 --session_id 140129 --phase wake --subset hdc --cv_folds 5 --cv -1 --ncvx 3 --batch_size 5000 --max_epochs 3000 --bin_size 160 --likelihood U-ec-3 --mapping svgp-64 --x_mode hd-omega-speed-x-y-time --lr 1e-2 --jitter 1e-5 --gpu 0 &

python3 HDC.py --checkpoint_dir /scratches/ramanujan_2/dl543/HDC_PartIII/checkpoint/ --data_path /scratches/ramanujan_2/dl543/HDC_PartIII/ --mouse_id Mouse25 --session_id 140129 --phase wake --subset nonhdc --cv_folds 5 --cv -1 --ncvx 3 --batch_size 1000 --max_epochs 3000 --bin_size 160 --likelihood U-ec-3 --mapping svgp-64 --x_mode hd-omega-speed-x-y-time --lr 1e-2 --jitter 1e-5 --gpu 0 &





python3 HDC.py --checkpoint_dir /scratches/ramanujan_2/dl543/HDC_PartIII/checkpoint/ --data_path /scratches/ramanujan_2/dl543/HDC_PartIII/ --mouse_id Mouse24 --session_id 131217 --phase wake --subset hdc --cv_folds 5 --cv -1 --ncvx 3 --batch_size 5000 --max_epochs 3000 --bin_size 160 --likelihood U-ec-3 --mapping svgp-64 --x_mode hd-omega-speed-x-y-time --lr 1e-2 --jitter 1e-5 --gpu 0 &

python3 HDC.py --checkpoint_dir /scratches/ramanujan_2/dl543/HDC_PartIII/checkpoint/ --data_path /scratches/ramanujan_2/dl543/HDC_PartIII/ --mouse_id Mouse24 --session_id 131217 --phase wake --subset nonhdc --cv_folds 5 --cv -1 --ncvx 3 --batch_size 1000 --max_epochs 3000 --bin_size 160 --likelihood U-ec-3 --mapping svgp-64 --x_mode hd-omega-speed-x-y-time --lr 1e-2 --jitter 1e-5 --gpu 0 &

python3 HDC.py --checkpoint_dir /scratches/ramanujan_2/dl543/HDC_PartIII/checkpoint/ --data_path /scratches/ramanujan_2/dl543/HDC_PartIII/ --mouse_id Mouse24 --session_id 131218 --phase wake --subset hdc --cv_folds 5 --cv -1 --ncvx 3 --batch_size 5000 --max_epochs 3000 --bin_size 160 --likelihood U-ec-3 --mapping svgp-64 --x_mode hd-omega-speed-x-y-time --lr 1e-2 --jitter 1e-5 --gpu 1 &

python3 HDC.py --checkpoint_dir /scratches/ramanujan_2/dl543/HDC_PartIII/checkpoint/ --data_path /scratches/ramanujan_2/dl543/HDC_PartIII/ --mouse_id Mouse24 --session_id 131218 --phase wake --subset nonhdc --cv_folds 5 --cv -1 --ncvx 3 --batch_size 1000 --max_epochs 3000 --bin_size 160 --likelihood U-ec-3 --mapping svgp-64 --x_mode hd-omega-speed-x-y-time --lr 1e-2 --jitter 1e-5 --gpu 1 &

