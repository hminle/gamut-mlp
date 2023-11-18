# echo "Run clipping method"
# python main_run.py experiment=exp_gma_cvpr method=clip

# echo "Run soft-clipping method"
# HYDRA_FULL_ERROR=1 python main_run.py experiment=exp_gma_cvpr method=soft_clip pipeline.stop_idx=1

# echo "Run MLP tiny"
# python main_run.py experiment=exp_gma_cvpr \
# method=mlp_tiny_cudnn \
# method.n_neurons=32 \
# method.is_trained=False \
# method.retrain=True \
# method.method_name=mlp_tiny_cudnn32_step10ksam50ogsam5 \
# method.sample=50 \
# method.og_sample=5 \
# method.gpus=3


# echo "meta_tiny32_innerstep10ksam50ogsam5_nsteps1200"
# python main_run.py experiment=exp_gma_cvpr \
# method=mlp_tiny_cudnn \
# method.n_neurons=32 \
# method.is_trained=True \
# method.retrain=True \
# method.method_name=meta_tiny32_innerstep10ksam50ogsam5_nsteps1200 \
# method.pretrained_model=/home/hminle/gitrepo/gma-mlp/pretrained_models/meta_tinycudnn32_metaep3_innersteps10k.pt \
# method.n_steps=1200 \
# method.sample=50 \
# method.og_sample=5 \
# method.gpus=0
