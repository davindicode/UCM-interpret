#!/bin/bash

cd ./scripts/



python3 HDC.py --session_id 12-120806 --phase wake --cv_folds 5 --cv 0 1 2 3 4 -1 --ncvx 2 --batch_size 10000 --bin_size 10 --likelihood Uqd3 --mapping svgp64 --x_mode hd-w-s-pos-t --lr 1e-2 --jitter 1e-5 --gpu 0 &
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


python3 HDC.py --session_id 12-120806 --phase wake --cv_folds 5 --cv 0 1 2 3 4 -1 --ncvx 2 --batch_size 5000 --bin_size 110 --likelihood Uqd3 --mapping svgp64 --x_mode hd-w-s-pos-t --lr 1e-2 --jitter 1e-5 --gpu 0 &
BACK_PID1=$!

python3 HDC.py --session_id 12-120806 --phase wake --cv_folds 5 --cv 0 1 2 3 4 -1 --ncvx 2 --batch_size 4000 --bin_size 130 --likelihood Uqd3 --mapping svgp64 --x_mode hd-w-s-pos-t --lr 1e-2 --jitter 1e-5 --gpu 0 &
BACK_PID2=$!

python3 HDC.py --session_id 12-120806 --phase wake --cv_folds 5 --cv 0 1 2 3 4 -1 --ncvx 2 --batch_size 3000 --bin_size 150 --likelihood Uqd3 --mapping svgp64 --x_mode hd-w-s-pos-t --lr 1e-2 --jitter 1e-5 --gpu 1 &
BACK_PID3=$!

python3 HDC.py --session_id 12-120806 --phase wake --cv_folds 5 --cv 0 1 2 3 4 -1 --ncvx 2 --batch_size 2000 --bin_size 170 --likelihood Uqd3 --mapping svgp64 --x_mode hd-w-s-pos-t --lr 1e-2 --jitter 1e-5 --gpu 1 &
BACK_PID4=$!

python3 HDC.py --session_id 12-120806 --phase wake --cv_folds 5 --cv 0 1 2 3 4 -1 --ncvx 2 --batch_size 1000 --bin_size 190 --likelihood Uqd3 --mapping svgp64 --x_mode hd-w-s-pos-t --lr 1e-2 --jitter 1e-5 --gpu 1 &
BACK_PID5=$!

wait $BACK_PID1
wait $BACK_PID2
wait $BACK_PID3
wait $BACK_PID4
wait $BACK_PID5

echo "Block done."
exit 1


python3 HDC.py --session_id 12-120806 --phase wake --cv_folds 5 --cv 0 1 2 3 4 -1 --ncvx 2 --batch_size 120000 --bin_size 1 --likelihood IPPexp --mapping svgp64 --x_mode hd-w-s-pos-t --jitter 1e-5 --lr 1e-2 --gpu 0

python3 HDC.py --session_id 12-120806 --phase wake --cv_folds 5 --cv 0 1 2 3 4 -1 --ncvx 2 --batch_size 130000 --bin_size 1 --likelihood IGexp --mapping svgp64 --x_mode hd-w-s-pos-t --lr 1e-2 --jitter 1e-5 --gpu 1

python3 HDC.py --session_id 12-120806 --phase wake --cv_folds 5 --cv 0 1 2 3 4 -1 --ncvx 2 --batch_size 130000 --bin_size 1 --likelihood IIGexp --mapping svgp64 --x_mode hd-w-s-pos-t --lr 1e-2 --jitter 1e-5 --gpu 0

python3 HDC.py --session_id 12-120806 --phase wake --cv_folds 5 --cv 0 1 2 3 4 -1 --ncvx 2 --batch_size 130000 --bin_size 1 --likelihood LNexp --mapping svgp64 --x_mode hd-w-s-pos-t --lr 1e-2 --jitter 1e-5 --gpu 1




### LVM ###
python3 HDC.py --session_id 12-120806 --phase wake --cv_folds 5 --cv 0 1 2 3 4 -1 --ncvx 3 --batch_size 12000 --bin_size 160 --likelihood Uqd3 --mapping svgp72 --x_mode hd-w-s-pos-t --z_mode R1 --lr 1e-2 --lr_2 1e-3 --jitter 1e-5 --gpu 0 &

python3 HDC.py --session_id 12-120806 --phase wake --cv_folds 5 --cv 0 1 2 3 4 -1 --ncvx 3 --batch_size 10000 --bin_size 160 --likelihood Uqd3 --mapping svgp80 --x_mode hd-w-s-pos-t --z_mode R2 --lr 1e-2 --lr_2 1e-3 --jitter 1e-5 --gpu 1 &

python3 HDC.py --session_id 12-120806 --phase wake --cv_folds 5 --cv 0 1 2 3 4 -1 --ncvx 3 --batch_size 10000 --bin_size 160 --likelihood Uqd3 --mapping svgp88 --x_mode hd-w-s-pos-t --z_mode R3 --lr 1e-2 --lr_2 1e-3 --jitter 1e-5 --gpu 0 &

python3 HDC.py --session_id 12-120806 --phase wake --cv_folds 5 --cv 0 1 2 3 4 -1 --ncvx 3 --batch_size 8000 --bin_size 160 --likelihood Uqd3 --mapping svgp96 --x_mode hd-w-s-pos-t --z_mode R4 --lr 1e-2 --lr_2 1e-3 --jitter 1e-5 --gpu 1 &

python3 HDC.py --session_id 12-120806 --phase wake --cv_folds 5 --cv 0 1 2 3 4 -1 --ncvx 3 --batch_size 6000 --bin_size 160 --likelihood Uqd3 --mapping svgp104 --x_mode hd-w-s-pos-t --z_mode R5 --lr 1e-2 --lr_2 1e-3 --jitter 1e-5 --gpu 0 & # TODO




# 28-140313
python3 HDC.py --session_id 28-140313 --phase wake --cv_folds 5 --cv 0 1 2 3 4 -1 --ncvx 2 --batch_size 2500 --bin_size 100 --likelihood Uqd3 --mapping svgp64 --x_mode hd-w-s-pos-t --lr 1e-2 --jitter 1e-4 --gpu 0 &

python3 HDC.py --session_id 28-140313 --phase wake --cv_folds 5 --cv 0 1 2 3 4 -1 --ncvx 2 --batch_size 1500 --bin_size 200 --likelihood Uqd3 --mapping svgp64 --x_mode hd-w-s-pos-t --lr 1e-2 --jitter 1e-4 --gpu 1 &

python3 HDC.py --session_id 28-140313 --phase wake --cv_folds 5 --cv 0 1 2 3 4 -1 --ncvx 2 --batch_size 1000 --bin_size 300 --likelihood Uqd3 --mapping svgp64 --x_mode hd-w-s-pos-t --lr 1e-2 --jitter 1e-4 --gpu 0 &

python3 HDC.py --session_id 28-140313 --phase wake --cv_folds 5 --cv 0 1 2 3 4 -1 --ncvx 2 --batch_size 1000 --bin_size 400 --likelihood Uqd3 --mapping svgp64 --x_mode hd-w-s-pos-t --lr 1e-2 --jitter 1e-4 --gpu 1 &




python3 HDC.py --session_id 28-140313 --phase wake --cv_folds 5 --cv 0 1 2 3 4 -1 --ncvx 2 --batch_size 6000 --bin_size 20 --likelihood Uqd3 --mapping svgp64 --x_mode hd-w-s-pos-t --lr 1e-2 --jitter 1e-4 --gpu 0 &

python3 HDC.py --session_id 28-140313 --phase wake --cv_folds 5 --cv 0 1 2 3 4 -1 --ncvx 2 --batch_size 5000 --bin_size 40 --likelihood Uqd3 --mapping svgp64 --x_mode hd-w-s-pos-t --lr 1e-2 --jitter 1e-4 --gpu 0 &

python3 HDC.py --session_id 28-140313 --phase wake --cv_folds 5 --cv 0 1 2 3 4 -1 --ncvx 2 --batch_size 4000 --bin_size 60 --likelihood Uqd3 --mapping svgp64 --x_mode hd-w-s-pos-t --lr 1e-2 --jitter 1e-4 --gpu 1 &

python3 HDC.py --session_id 28-140313 --phase wake --cv_folds 5 --cv 0 1 2 3 4 -1 --ncvx 2 --batch_size 3000 --bin_size 80 --likelihood Uqd3 --mapping svgp64 --x_mode hd-w-s-pos-t --lr 1e-2 --jitter 1e-4 --gpu 1 &



python3 HDC.py --session_id 28-140313 --phase wake --cv_folds 5 --cv 0 1 2 3 4 -1 --ncvx 2 --batch_size 2000 --bin_size 120 --likelihood Uqd3 --mapping svgp64 --x_mode hd-w-s-pos-t --lr 1e-2 --jitter 1e-4 --gpu 0 &

python3 HDC.py --session_id 28-140313 --phase wake --cv_folds 5 --cv 0 1 2 3 4 -1 --ncvx 2 --batch_size 1500 --bin_size 140 --likelihood Uqd3 --mapping svgp64 --x_mode hd-w-s-pos-t --lr 1e-2 --jitter 1e-4 --gpu 0 &

python3 HDC.py --session_id 28-140313 --phase wake --cv_folds 5 --cv 0 1 2 3 4 -1 --ncvx 2 --batch_size 1500 --bin_size 160 --likelihood Uqd3 --mapping svgp64 --x_mode hd-w-s-pos-t --lr 1e-2 --jitter 1e-4 --gpu 1 &

python3 HDC.py --session_id 28-140313 --phase wake --cv_folds 5 --cv 0 1 2 3 4 -1 --ncvx 2 --batch_size 1500 --bin_size 180 --likelihood Uqd3 --mapping svgp64 --x_mode hd-w-s-pos-t --lr 1e-2 --jitter 1e-4 --gpu 1 &



python3 HDC.py --session_id 28-140313 --phase wake --cv_folds 5 --cv 0 1 2 3 4 -1 --ncvx 2 --batch_size 20000 --bin_size 10 --likelihood Uqd3 --mapping svgp64 --x_mode hd-w-s-pos-t --lr 1e-2 --jitter 1e-4 --gpu 0 &

python3 HDC.py --session_id 28-140313 --phase wake --cv_folds 5 --cv 0 1 2 3 4 -1 --ncvx 2 --batch_size 6000 --bin_size 30 --likelihood Uqd3 --mapping svgp64 --x_mode hd-w-s-pos-t --lr 1e-2 --jitter 1e-4 --gpu 0 &

python3 HDC.py --session_id 28-140313 --phase wake --cv_folds 5 --cv 0 1 2 3 4 -1 --ncvx 2 --batch_size 12000 --bin_size 50 --likelihood Uqd3 --mapping svgp64 --x_mode hd-w-s-pos-t --lr 1e-2 --jitter 1e-4 --gpu 1 &

python3 HDC.py --session_id 28-140313 --phase wake --cv_folds 5 --cv 0 1 2 3 4 -1 --ncvx 2 --batch_size 3000 --bin_size 70 --likelihood Uqd3 --mapping svgp64 --x_mode hd-w-s-pos-t --lr 1e-2 --jitter 1e-4 --gpu 1 &



python3 HDC.py --session_id 28-140313 --phase wake --cv_folds 5 --cv 0 1 2 3 4 -1 --ncvx 2 --batch_size 8000 --bin_size 90 --likelihood Uqd3 --mapping svgp64 --x_mode hd-w-s-pos-t --lr 1e-2 --jitter 1e-4 --gpu 0 &

python3 HDC.py --session_id 28-140313 --phase wake --cv_folds 5 --cv 0 1 2 3 4 -1 --ncvx 2 --batch_size 1500 --bin_size 500 --likelihood Uqd3 --mapping svgp64 --x_mode hd-w-s-pos-t --lr 1e-2 --jitter 1e-4 --gpu 1 &





### IPP ###
python3 HDC.py --session_id 28-140313 --phase wake --cv_folds 5 --cv 0 1 2 3 4 -1 --ncvx 2 --batch_size 50000 --bin_size 20 --likelihood IPexp --mapping svgp64 --x_mode hd-w-s-pos-t --lr 1e-2 --jitter 1e-5 --gpu 0 &

python3 HDC.py --session_id 28-140313 --phase wake --cv_folds 5 --cv 0 1 2 3 4 -1 --ncvx 2 --batch_size 40000 --bin_size 40 --likelihood IPexp --mapping svgp64 --x_mode hd-w-s-pos-t --lr 1e-2 --jitter 1e-5 --gpu 0 &

python3 HDC.py --session_id 28-140313 --phase wake --cv_folds 5 --cv 0 1 2 3 4 -1 --ncvx 2 --batch_size 30000 --bin_size 60 --likelihood IPexp --mapping svgp64 --x_mode hd-w-s-pos-t --lr 1e-2 --jitter 1e-5 --gpu 1 &

python3 HDC.py --session_id 28-140313 --phase wake --cv_folds 5 --cv 0 1 2 3 4 -1 --ncvx 2 --batch_size 20000 --bin_size 80 --likelihood IPexp --mapping svgp64 --x_mode hd-w-s-pos-t --lr 1e-2 --jitter 1e-5 --gpu 1 &



python3 HDC.py --session_id 28-140313 --phase wake --cv_folds 5 --cv 0 1 2 3 4 -1 --ncvx 2 --batch_size 16000 --bin_size 100 --likelihood IPexp --mapping svgp64 --x_mode hd-w-s-pos-t --lr 1e-2 --jitter 1e-5 --gpu 0 &

python3 HDC.py --session_id 28-140313 --phase wake --cv_folds 5 --cv 0 1 2 3 4 -1 --ncvx 2 --batch_size 14000 --bin_size 200 --likelihood IPexp --mapping svgp64 --x_mode hd-w-s-pos-t --lr 1e-2 --jitter 1e-5 --gpu 0 &

python3 HDC.py --session_id 28-140313 --phase wake --cv_folds 5 --cv 0 1 2 3 4 -1 --ncvx 2 --batch_size 12000 --bin_size 300 --likelihood IPexp --mapping svgp64 --x_mode hd-w-s-pos-t --lr 1e-2 --jitter 1e-5 --gpu 1 &

python3 HDC.py --session_id 28-140313 --phase wake --cv_folds 5 --cv 0 1 2 3 4 -1 --ncvx 2 --batch_size 10000 --bin_size 400 --likelihood IPexp --mapping svgp64 --x_mode hd-w-s-pos-t --lr 1e-2 --jitter 1e-5 --gpu 1 &

python3 HDC.py --session_id 28-140313 --phase wake --cv_folds 5 --cv 0 1 2 3 4 -1 --ncvx 2 --batch_size 8000 --bin_size 500 --likelihood IPexp --mapping svgp64 --x_mode hd-w-s-pos-t --lr 1e-2 --jitter 1e-5 --gpu 0 &






# 12-120806
python3 HDC.py --session_id 12-120806 --phase wake --cv_folds 5 --cv 0 1 2 3 4 -1 --ncvx 2 --batch_size 16000 --bin_size 100 --likelihood IPexp --mapping svgp64 --x_mode hd-w-s-pos-t --lr 1e-2 --jitter 1e-5 --gpu 0 &

python3 HDC.py --session_id 12-120806 --phase wake --cv_folds 5 --cv 0 1 2 3 4 -1 --ncvx 2 --batch_size 14000 --bin_size 200 --likelihood IPexp --mapping svgp64 --x_mode hd-w-s-pos-t --lr 1e-2 --jitter 1e-5 --gpu 0 &

python3 HDC.py --session_id 12-120806 --phase wake --cv_folds 5 --cv 0 1 2 3 4 -1 --ncvx 2 --batch_size 12000 --bin_size 300 --likelihood IPexp --mapping svgp64 --x_mode hd-w-s-pos-t --lr 1e-2 --jitter 1e-5 --gpu 1 &

python3 HDC.py --session_id 12-120806 --phase wake --cv_folds 5 --cv 0 1 2 3 4 -1 --ncvx 2 --batch_size 10000 --bin_size 400 --likelihood IPexp --mapping svgp64 --x_mode hd-w-s-pos-t --lr 1e-2 --jitter 1e-5 --gpu 1 &

python3 HDC.py --session_id 12-120806 --phase wake --cv_folds 5 --cv 0 1 2 3 4 -1 --ncvx 2 --batch_size 8000 --bin_size 500 --likelihood IPexp --mapping svgp64 --x_mode hd-w-s-pos-t --lr 1e-2 --jitter 1e-5 --gpu 0 &


python3 HDC.py --session_id 12-120806 --phase wake --cv_folds 5 --cv 0 1 2 3 4 -1 --ncvx 2 --batch_size 50000 --bin_size 10 --likelihood IPexp --mapping svgp64 --x_mode hd-w-s-pos-t --lr 1e-2 --jitter 1e-5 --gpu 0 &

python3 HDC.py --session_id 12-120806 --phase wake --cv_folds 5 --cv 0 1 2 3 4 -1 --ncvx 2 --batch_size 40000 --bin_size 30 --likelihood IPexp --mapping svgp64 --x_mode hd-w-s-pos-t --lr 1e-2 --jitter 1e-5 --gpu 0 &

python3 HDC.py --session_id 12-120806 --phase wake --cv_folds 5 --cv 0 1 2 3 4 -1 --ncvx 2 --batch_size 30000 --bin_size 50 --likelihood IPexp --mapping svgp64 --x_mode hd-w-s-pos-t --lr 1e-2 --jitter 1e-5 --gpu 1 &

python3 HDC.py --session_id 12-120806 --phase wake --cv_folds 5 --cv 0 1 2 3 4 -1 --ncvx 2 --batch_size 20000 --bin_size 70 --likelihood IPexp --mapping svgp64 --x_mode hd-w-s-pos-t --lr 1e-2 --jitter 1e-5 --gpu 1 &

python3 HDC.py --session_id 12-120806 --phase wake --cv_folds 5 --cv 0 1 2 3 4 -1 --ncvx 2 --batch_size 20000 --bin_size 90 --likelihood IPexp --mapping svgp64 --x_mode hd-w-s-pos-t --lr 1e-2 --jitter 1e-5 --gpu 1 &


python3 HDC.py --session_id 12-120806 --phase wake --cv_folds 5 --cv 0 1 2 3 4 -1 --ncvx 2 --batch_size 50000 --bin_size 20 --likelihood IPexp --mapping svgp64 --x_mode hd-w-s-pos-t --lr 1e-2 --jitter 1e-5 --gpu 1 &

python3 HDC.py --session_id 12-120806 --phase wake --cv_folds 5 --cv 0 1 2 3 4 -1 --ncvx 2 --batch_size 40000 --bin_size 40 --likelihood IPexp --mapping svgp64 --x_mode hd-w-s-pos-t --lr 1e-2 --jitter 1e-5 --gpu 1 &

python3 HDC.py --session_id 12-120806 --phase wake --cv_folds 5 --cv 0 1 2 3 4 -1 --ncvx 2 --batch_size 30000 --bin_size 60 --likelihood IPexp --mapping svgp64 --x_mode hd-w-s-pos-t --lr 1e-2 --jitter 1e-5 --gpu 1 &

python3 HDC.py --session_id 12-120806 --phase wake --cv_folds 5 --cv 0 1 2 3 4 -1 --ncvx 2 --batch_size 20000 --bin_size 80 --likelihood IPexp --mapping svgp64 --x_mode hd-w-s-pos-t --lr 1e-2 --jitter 1e-5 --gpu 1 &


python3 HDC.py --session_id 12-120806 --phase wake --cv_folds 5 --cv 0 1 2 3 4 -1 --ncvx 2 --batch_size 15000 --bin_size 120 --likelihood IPexp --mapping svgp64 --x_mode hd-w-s-pos-t --lr 1e-2 --jitter 1e-5 --gpu 1 &

python3 HDC.py --session_id 12-120806 --phase wake --cv_folds 5 --cv 0 1 2 3 4 -1 --ncvx 2 --batch_size 50000 --bin_size 140 --likelihood IPexp --mapping svgp64 --x_mode hd-w-s-pos-t --lr 1e-2 --jitter 1e-5 --gpu 1 &

python3 HDC.py --session_id 12-120806 --phase wake --cv_folds 5 --cv 0 1 2 3 4 -1 --ncvx 2 --batch_size 9000 --bin_size 160 --likelihood IPexp --mapping svgp64 --x_mode hd-w-s-pos-t --lr 1e-2 --jitter 1e-5 --gpu 0 &

python3 HDC.py --session_id 12-120806 --phase wake --cv_folds 5 --cv 0 1 2 3 4 -1 --ncvx 2 --batch_size 7000 --bin_size 180 --likelihood IPexp --mapping svgp64 --x_mode hd-w-s-pos-t --lr 1e-2 --jitter 1e-5 --gpu 0 &


python3 HDC.py --session_id 12-120806 --phase wake --cv_folds 5 --cv 0 1 2 3 4 -1 --ncvx 2 --batch_size 5000 --bin_size 220 --likelihood IPexp --mapping svgp64 --x_mode hd-w-s-pos-t --lr 1e-2 --jitter 1e-5 --gpu 0

python3 HDC.py --session_id 12-120806 --phase wake --cv_folds 5 --cv 0 1 2 3 4 -1 --ncvx 2 --batch_size 4000 --bin_size 240 --likelihood IPexp --mapping svgp64 --x_mode hd-w-s-pos-t --lr 1e-2 --jitter 1e-5 --gpu 0

python3 HDC.py --session_id 12-120806 --phase wake --cv_folds 5 --cv 0 1 2 3 4 -1 --ncvx 2 --batch_size 3000 --bin_size 260 --likelihood IPexp --mapping svgp64 --x_mode hd-w-s-pos-t --lr 1e-2 --jitter 1e-5 --gpu 0

python3 HDC.py --session_id 12-120806 --phase wake --cv_folds 5 --cv 0 1 2 3 4 -1 --ncvx 2 --batch_size 2000 --bin_size 280 --likelihood IPexp --mapping svgp64 --x_mode hd-w-s-pos-t --lr 1e-2 --jitter 1e-5 --gpu 0











python3 HDC.py --session_id 12-120806 --phase wake --cv_folds 5 --cv 0 1 2 3 4 -1 --ncvx 2 --batch_size 4000 --bin_size 100 --likelihood Uqd3 --mapping svgp64 --x_mode hd-w-s-pos-t --lr 1e-2 --jitter 1e-5 --gpu 0 &

python3 HDC.py --session_id 12-120806 --phase wake --cv_folds 5 --cv 0 1 2 3 4 -1 --ncvx 2 --batch_size 4000 --bin_size 200 --likelihood Uqd3 --mapping svgp64 --x_mode hd-w-s-pos-t --lr 1e-2 --jitter 1e-5 --gpu 0 &

python3 HDC.py --session_id 12-120806 --phase wake --cv_folds 5 --cv 0 1 2 3 4 -1 --ncvx 2 --batch_size 3000 --bin_size 300 --likelihood Uqd3 --mapping svgp64 --x_mode hd-w-s-pos-t --lr 1e-2 --jitter 1e-5 --gpu 0 &

python3 HDC.py --session_id 12-120806 --phase wake --cv_folds 5 --cv 0 1 2 3 4 -1 --ncvx 2 --batch_size 4000 --bin_size 400 --likelihood Uqd3 --mapping svgp64 --x_mode hd-w-s-pos-t --lr 1e-2 --jitter 1e-5 --gpu 1 &

python3 HDC.py --session_id 12-120806 --phase wake --cv_folds 5 --cv 0 1 2 3 4 -1 --ncvx 2 --batch_size 4000 --bin_size 500 --likelihood Uqd3 --mapping svgp64 --x_mode hd-w-s-pos-t --lr 1e-2 --jitter 1e-5 --gpu 1 &



python3 HDC.py --session_id 12-120806 --phase wake --cv_folds 5 --cv 0 1 2 3 4 -1 --ncvx 2 --batch_size 5000 --bin_size 120 --likelihood Uqd3 --mapping svgp64 --x_mode hd-w-s-pos-t --lr 1e-2 --jitter 1e-5 --gpu 1 &

python3 HDC.py --session_id 12-120806 --phase wake --cv_folds 5 --cv 0 1 2 3 4 -1 --ncvx 2 --batch_size 5000 --bin_size 140 --likelihood Uqd3 --mapping svgp64 --x_mode hd-w-s-pos-t --lr 1e-2 --jitter 1e-5 --gpu 0 &

python3 HDC.py --session_id 12-120806 --phase wake --cv_folds 5 --cv 0 1 2 3 4 -1 --ncvx 2 --batch_size 6000 --bin_size 160 --likelihood Uqd3 --mapping svgp64 --x_mode hd-w-s-pos-t --lr 1e-2 --jitter 1e-5 --gpu 1 &

python3 HDC.py --session_id 12-120806 --phase wake --cv_folds 5 --cv 0 1 2 3 4 -1 --ncvx 2 --batch_size 2000 --bin_size 180 --likelihood Uqd3 --mapping svgp64 --x_mode hd-w-s-pos-t --lr 1e-2 --jitter 1e-5 --gpu 1 &


python3 HDC.py --session_id 12-120806 --phase wake --cv_folds 5 --cv 0 1 2 3 4 -1 --ncvx 2 --batch_size 4000 --bin_size 220 --likelihood Uqd3 --mapping svgp64 --x_mode hd-w-s-pos-t --lr 1e-2 --jitter 1e-5 --gpu 0

python3 HDC.py --session_id 12-120806 --phase wake --cv_folds 5 --cv 0 1 2 3 4 -1 --ncvx 2 --batch_size 2000 --bin_size 240 --likelihood Uqd3 --mapping svgp64 --x_mode hd-w-s-pos-t --lr 1e-2 --jitter 1e-5 --gpu 0

python3 HDC.py --session_id 12-120806 --phase wake --cv_folds 5 --cv 0 1 2 3 4 -1 --ncvx 2 --batch_size 3000 --bin_size 260 --likelihood Uqd3 --mapping svgp64 --x_mode hd-w-s-pos-t --lr 1e-2 --jitter 1e-5 --gpu 1

python3 HDC.py --session_id 12-120806 --phase wake --cv_folds 5 --cv 0 1 2 3 4 -1 --ncvx 2 --batch_size 2000 --bin_size 280 --likelihood Uqd3 --mapping svgp64 --x_mode hd-w-s-pos-t --lr 1e-2 --jitter 1e-5 --gpu 1







python3 HDC.py --session_id 12-120806 --phase wake --cv_folds 5 --cv -1 --ncvx 2 --batch_size 120000 --bin_size 1 --likelihood IPPexp --mapping svgp64 --x_mode hd-w-s-pos-t --jitter 1e-5 --lr 1e-2 --gpu 0

python3 HDC.py --session_id 12-120806 --phase wake --cv_folds 5 --cv 0 --ncvx 2 --batch_size 120000 --bin_size 1 --likelihood IPPexp --mapping svgp64 --x_mode hd-w-s-pos-t --jitter 1e-5 --lr 1e-2 --gpu 0

python3 HDC.py --session_id 12-120806 --phase wake --cv_folds 5 --cv 1 --ncvx 2 --batch_size 120000 --bin_size 1 --likelihood IPPexp --mapping svgp64 --x_mode hd-w-s-pos-t --jitter 1e-5 --lr 1e-2 --gpu 1

python3 HDC.py --session_id 12-120806 --phase wake --cv_folds 5 --cv 2 --ncvx 2 --batch_size 120000 --bin_size 1 --likelihood IPPexp --mapping svgp64 --x_mode hd-w-s-pos-t --jitter 1e-5 --lr 1e-2 --gpu 0

python3 HDC.py --session_id 12-120806 --phase wake --cv_folds 5 --cv 3 --ncvx 2 --batch_size 120000 --bin_size 1 --likelihood IPPexp --mapping svgp64 --x_mode hd-w-s-pos-t --jitter 1e-5 --lr 1e-2 --gpu 1

python3 HDC.py --session_id 12-120806 --phase wake --cv_folds 5 --cv 0 1 2 3 4 -1 --ncvx 2 --batch_size 120000 --bin_size 1 --likelihood IPPexp --mapping svgp64 --x_mode hd-w-s-pos-t --jitter 1e-5 --lr 1e-2 --gpu 1




python3 HDC.py --session_id 28-140313 --phase wake --cv_folds 5 --cv 0 1 2 3 4 -1 --ncvx 2 --batch_size 4000 --bin_size 100 --likelihood Uqd3 --mapping svgp64 --x_mode hd-w-s-pos-t --lr 1e-2 --jitter 1e-5 --gpu 0

python3 HDC.py --session_id 28-140313 --phase wake --cv_folds 5 --cv 0 1 2 3 4 -1 --ncvx 2 --batch_size 4000 --bin_size 200 --likelihood Uqd3 --mapping svgp64 --x_mode hd-w-s-pos-t --lr 1e-2 --jitter 1e-5 --gpu 0

python3 HDC.py --session_id 28-140313 --phase wake --cv_folds 5 --cv 0 1 2 3 4 -1 --ncvx 2 --batch_size 3000 --bin_size 300 --likelihood Uqd3 --mapping svgp64 --x_mode hd-w-s-pos-t --lr 1e-2 --jitter 1e-5 --gpu 0

python3 HDC.py --session_id 28-140313 --phase wake --cv_folds 5 --cv 0 1 2 3 4 -1 --ncvx 2 --batch_size 4000 --bin_size 400 --likelihood Uqd3 --mapping svgp64 --x_mode hd-w-s-pos-t --lr 1e-2 --jitter 1e-5 --gpu 1

python3 HDC.py --session_id 28-140313 --phase wake --cv_folds 5 --cv 0 1 2 3 4 -1 --ncvx 2 --batch_size 4000 --bin_size 500 --likelihood Uqd3 --mapping svgp64 --x_mode hd-w-s-pos-t --lr 1e-2 --jitter 1e-5 --gpu 1



python3 HDC.py --session_id 12-120806 --phase wake --cv_folds 5 --cv 0 1 2 3 4 -1 --ncvx 2 --batch_size 4000 --bin_size 100 --likelihood hCMPexp --mapping svgp64 --x_mode hd-w-s-pos-t --lr 1e-2 --jitter 1e-5 --gpu 1






