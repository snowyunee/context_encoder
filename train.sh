#!/bin/bash
#python main.py --optimizer=sgd --dataset=CelebA --use_gpu=True --mask_center_path=./mask/Mask_Glasses_fix01.png --mask_overlap_path=./mask/Mask_Glasses_fix04.png --load_path=CelebA_0822_090607

# step 0
python main.py --optimizer=adam --dataset=CelebA --use_gpu=True --mask_dir=./mask_mss --max_step=100010 --save_step=100 --d_lr=0.0002 --g_lr=0.002 --init_lr=True --load_path=CelebA_0822_174033

# step 1
#python main.py --optimizer=adam --dataset=CelebA --use_gpu=True --mask_dir=./mask_mss --max_step=20010 --save_step=100 --d_lr=0.0001 --g_lr=0.001 --load_path=CelebA_0822_174033

# step 2
#python main.py --optimizer=adam --dataset=CelebA --use_gpu=True --mask_dir=./mask_mss --max_step=20010 --save_step=100 --d_lr=0.00005 --g_lr=0.0005 --load_path=CelebA_0822_174033

# step 3
#python main.py --optimizer=adam --dataset=CelebA --use_gpu=True --mask_dir=./mask_mss --max_step=20010 --save_step=100 --d_lr=0.000025 --g_lr=0.00025 --load_path=CelebA_0822_174033

# step 4
#python main.py --optimizer=adam --dataset=CelebA --use_gpu=True --mask_dir=./mask_mss --max_step=20010 --save_step=100 --d_lr=0.00002 --g_lr=0.000125 --load_path=CelebA_0822_174033

