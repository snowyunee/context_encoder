#!/bin/bash
#python main.py --dataset=CelebA --use_gpu=True --is_train=False --split=test --load_path=CelebA_0614_232948


#python main.py --dataset=CelebA --load_path=CelebA_0614_232948 --use_gpu=True --is_train=False --split=hr --batch_size=2
python main.py --dataset=CelebA --load_path=CelebA_0614_232948 --use_gpu=True --is_train=False --split=train --batch_size=2


