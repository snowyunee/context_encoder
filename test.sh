#!/bin/bash
#python main.py --dataset=CelebA --use_gpu=True --is_train=False --split=test --load_path=CelebA_0614_232948


#python main.py --dataset=CelebA --use_gpu=True --is_train=False --split=hr --batch_size=2 --mask_center_path=Mask_Glasses_fix01.png --mask_overlap_path=Mask_Glasses_fix04.png --load_path=CelebA_0614_181757

python main.py --dataset=CelebA --use_gpu=True --is_train=False --split=test --batch_size=2 --mask_center_path=./mask/Mask_Glasses_fix01.png --mask_overlap_path=./mask/Mask_Glasses_fix04.png --load_path=CelebA_0822_090607


