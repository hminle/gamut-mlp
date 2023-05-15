# GamutMLP - A Lightweight MLP for Color Loss Recovery (CVPR 2023)

*[Hoang M. Le](https://hminle.com)*<sup>1</sup>, *[Brian Price](https://www.brianpricephd.com/)*<sup>2</sup>, *[Scott Cohen](https://research.adobe.com/person/scott-cohen/)*<sup>2</sup> and *[Michael S. Brown](http://www.cse.yorku.ca/~mbrown/)*<sup>1</sup>

<sup>1</sup>York University

<sup>2</sup>Adobe Research


**This software is provided for research purposes only and CANNOT be used for commercial purposes.**

## Project Website: https://gamut-mlp.github.io/

## BibTex

Please cite us if you use this code or our dataset:

```
@inproceedings{hoangle2023GamutMLP,
    author    = {Le, Hoang M. and Price, Brian and Cohen, Scott and Brown, Michael S.},
    title     = {GamutMLP: A Lightweight MLP for Color Loss Recovery},
    booktitle = {CVPR},
    year      = {2023},
}
```

## Dataset:
- Test dataset: [Download 200 16-bit ProPhoto images](https://ln5.sync.com/dl/7cd8aa110/awnfsd8r-tfmkrcyg-x85j6w6s-d5qfcea7)
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
- Note: see `configs` for more settings
- Checkout our running scripts for more examples of other baseline methods: `scripts/cvpr_2023.sh`
