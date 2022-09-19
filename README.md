### SPU-Net: *Self-Supervised Point Cloud Upsampling by Coarse-to-Fine Reconstruction with Self-Projection Optimization*
Created by Xinhai Liu, Zhizhong Han, <a href="https://yushen-liu.github.io/" target="_blank">Yu-Shen Liu</a>, Matthias Zwicker.

### Usage
For training, the cmd example: 

        python train_spu-net.py --gpu gpu_index --log_dir log/spu_00012_git

For evaluating, the cmd example:


        python evaluate.py --gt data/test/groundtruth/ --pred log/spu_00012_git/output/


The training data from PU-GAN:

path: data/test

### Citation
If you find our work useful in your research, please consider citing:


        @article{liu2022spu,
          title={Spu-net: Self-supervised point cloud upsampling by coarse-to-fine reconstruction with self-projection optimization},
          author={Liu, Xinhai and Liu, Xinchen and Liu, Yu-Shen and Han, Zhizhong},
          journal={IEEE Transactions on Image Processing},
          volume={31},
          pages={4213--4226},
          year={2022},
          publisher={IEEE}
        }

