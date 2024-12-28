import torch
from torch.utils.data import Dataset
import os
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args, GroupParams, Namespace
from gaussian_renderer import GaussianModel
from scene import Scene

class HarmonizationDataset(Dataset):
    def __init__(self, sh_deg, scene_dir, iteration=-1, transform=None):
        self.gs_scenes = [os.path.join(scene_dir, f) for f in os.listdir(scene_dir) if os.path.isdir(os.path.join(scene_dir, f))]
        self.sh_deg = sh_deg
        self.iteration = iteration
        self.transform=transform

    def __getitem__(self, index):
        parser = ArgumentParser(description="Testing script parameters")
        model = ModelParams(parser, sentinel=True)
        pipeline = PipelineParams(parser)
        custom_args = {
            "model_path": self.gs_scenes[index],
            "depths": "",
            "train_test_exp": False
        }
        args = get_combined_args(parser, custom_args)
        model.extract(args)
        pipeline.extract(args)

        gaussians = GaussianModel(self.sh_deg)
        scene = Scene(args, gaussians, self.iteration, shuffle=False)

        return scene

    def __len__(self):
        return len(self.gs_scenes)