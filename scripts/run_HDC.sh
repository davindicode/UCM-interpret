#!/bin/bash



echo "Running block of 5 models..."

python3 HDC.py --data_path /scratches/ramanujan_2/dl543/HDC_PartIII/ --mouse_id Mouse12 --session_id 120806 --phase wake --subset hdc --cv_folds 5 --cv 0 1 2 3 4 -1 --ncvx 2 --batch_size 10000 --max_epochs 2 --bin_size 160 --likelihood U-ec-3 --mapping svgp-64 --x_mode hd-omega-speed-x-y-time --lr 1e-2 --jitter 1e-5 --gpu 0 &
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
