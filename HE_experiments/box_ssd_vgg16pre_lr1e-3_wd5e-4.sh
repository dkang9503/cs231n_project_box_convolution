python3 ../ssd.pytorch/train.py\
	--arch box_ssd\
	--dataset VOC\
	--dataset_root ~/Project/Code/data/VOCdevkit/\
	--save_folder ~/Project/Code/weights/\
	--lr 1e-3\
	--weight_decay 5e-4\
	--visdom True
