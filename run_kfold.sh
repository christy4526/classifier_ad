#!/bin/bash
for i in `seq 0 1 9`
do
  python train.py plane_denoise_deskull --pretrain --running_k $i --devices $i --labels AD,NL --denoise --deskull &
done
