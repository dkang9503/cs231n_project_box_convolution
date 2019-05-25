python3 ../train.py \
	--arch resnet18 \
	--lr 0.05 \
	--weight-decay 0.0001 \
	--momentum 0.9 \
	--epochs 50 \
	--batch-size 128 \
	--print-freq 100 \
	~/Project/Code/data/ImageNet/tiny-imagenet-200
