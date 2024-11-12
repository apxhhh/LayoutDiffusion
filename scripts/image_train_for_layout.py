"""
Train a diffusion model on images.
"""

import argparse
import os
import torch.optim as optim

import torch.distributed as dist
from omegaconf import OmegaConf

from layout_diffusion import dist_util, logger
from layout_diffusion.train_util import TrainLoop
from layout_diffusion.util import loopy
from layout_diffusion.layout_diffusion_unet import build_model
from layout_diffusion.resample import build_schedule_sampler
from layout_diffusion.dataset.data_loader import build_loaders
from layout_diffusion.respace import build_diffusion





def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, default='./configs/LayoutDiffusion-v1.yaml')
    known_args, unknown_args = parser.parse_known_args()

    known_args = OmegaConf.create(known_args.__dict__)
    cfg = OmegaConf.merge(OmegaConf.load(known_args.config_file), known_args)
    if unknown_args:
        unknown_args = OmegaConf.from_dotlist(unknown_args)
        cfg = OmegaConf.merge(cfg, unknown_args)
    print(OmegaConf.to_yaml(cfg))

    local_rank = int(os.environ["LOCAL_RANK"])
    dist_util.setup_dist(local_rank=local_rank)
    logger.configure(dir=cfg.train.log_dir)
    logger.log(f'current rank == {dist.get_rank()}, total_num = {dist.get_world_size()}, \n, {cfg}')

    logger.log("creating model...")
    model = build_model(cfg)
    model.to(dist_util.dev())
    print(model)

    logger.log("creating diffusion...")
    diffusion = build_diffusion(cfg)

    logger.log("creating schedule sampler...")
    schedule_sampler = build_schedule_sampler(cfg, diffusion)

    logger.log("creating data loader...")
    train_loader = build_loaders(cfg, mode='train')

    # Initialize the optimizer
    optimizer = optim.Adam(model.parameters(), lr=cfg.train.learning_rate)
    
    # Initialize the learning rate scheduler
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    logger.log("training...")
    trainer = TrainLoop(
        model=model,
        diffusion=diffusion,
        schedule_sampler=schedule_sampler,
        data=loopy(train_loader),
        batch_size=cfg.data.parameters.train.batch_size,
        lr_scheduler=lr_scheduler,  # Pass the scheduler here
        **cfg.train
    )
    trainer.run_loop()


if __name__ == "__main__":
    main()
