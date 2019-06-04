python3 ../ssd.pytorch/train.py\
	--dataset VOC\
	--dataset_root ~/Project/Code/data/VOCdevkit/\
	--save_folder ~/Project/Code/weights/\
	--lr 1e-3\
	--weight_decay 5e-3\
	--visdom True\
	--subset_size 0.1
