# GamutMLP - A Lightweight MLP for Color Loss Recovery (CVPR 2023)

*[Hoang M. Le](https://hminle.com)*<sup>1</sup>, *[Brian Price](https://www.brianpricephd.com/)*<sup>2</sup>, *[Scott Cohen](https://research.adobe.com/person/scott-cohen/)*<sup>2</sup> and *[Michael S. Brown](http://www.cse.yorku.ca/~mbrown/)*<sup>1</sup>

<sup>1</sup>York University

<sup>2</sup>Adobe Research


**This software is provided for research purposes only and CANNOT be used for commercial purposes.**

## Project Website: https://gamut-mlp.github.io/

## BibTex

Please cite us if you use this code or our dataset:

```
@InProceedings{Le_2023_CVPR,
    author    = {Le, Hoang M. and Price, Brian and Cohen, Scott and Brown, Michael S.},
    title     = {GamutMLP: A Lightweight MLP for Color Loss Recovery},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2023},
    pages     = {18268-18277}
}
```

## Dataset:
- Test dataset: [Download 200 16-bit ProPhoto images](https://ln5.sync.com/dl/f192fec40/xx4z2biv-8sjhcmjg-5yhqh8y2-87bvn9xy)
- Train dataset: [Download 2000 512x512 16-bit ProPhoto images](https://ln5.sync.com/dl/b9fae1a30/cnfhh9a7-2r87gipm-c6b5u92t-47u8cjy7)

## Code:
Our source code is for PyTorch platforms. *There is no guarantee that the trained models produce EXACTLY the same results, but it should be equivalent.*

### Setup environment:
- We provide `environment.yml` for conda, which can be installed with: `conda env create -f environment.yml`
- NOTE: we only test with Linux system.

### Run experiment:
- After downloading the test dataset `prophoto_full_16b`, you should set the `data_root` in `configs/config.yaml` with your own path.
- In `configs/dataset/prophoto_full_16b.yaml`, `dataset_name` should be the name of test dataset folder, which is originally `prophoto_full_16b`.
- How to run the fast MLP:
```bash
echo "Run MLP tiny"
python main_run.py experiment=exp_gma_cvpr \
method=mlp_tiny_cudnn \
method.n_neurons=32 \
method.is_trained=False \
method.retrain=True \
method.method_name=mlp_tiny_cudnn32_step10ksam50ogsam5 \
method.sample=50 \
method.og_sample=5 \
method.gpus=3
```

- Train the meta-init MLP:
```bash
python main_run.py experiment=exp_gma_train_meta \
pipeline.meta_inner_steps=10000 \
pipeline.meta_epoch=3
```

- Run the meta-init fast MLP:
```bash
python main_run.py experiment=exp_gma_cvpr \
method=mlp_tiny_cudnn \
method.n_neurons=32 \
method.is_trained=True \
method.retrain=True \
method.method_name=meta_tiny32 \
method.pretrained_model=<replace with project's absolute path here>/pretrained_models/meta_tinycudnn32_metaep3_innersteps10k.pt \
method.n_steps=1200 \
method.sample=50 \
method.og_sample=5 \
method.gpus=0
```


- Note: see `configs` for more settings
- Checkout our running scripts for more examples of other baseline methods: `scripts/cvpr_2023.sh`
