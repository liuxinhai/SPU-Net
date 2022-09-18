# SPU-Net
SPU-Net: Self-supervised Point Cloud Upsampling by Coarse-to-Fine Reconstruction with Self-Projection Optimization

For training, the cmd example: 
python train_spu-net.py --gpu 1 --log_dir log/spu_00012_git

For evaluating, the cmd example:
python evaluate.py --gt data/test/groundtruth/ --pred log/spu_00012_git/output/

The training data from PU-GAN:
path: data/test
