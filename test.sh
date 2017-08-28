#!/bin/bash
python main.py --dataset=CelebA --use_gpu=True --is_train=False --optimizer=sgd --split=mss_with_glasses --batch_size=2 --mask_dir=./mask_mss --load_path=CelebA_0822_174033


