import torch
from torch.utils.data import Dataset
import os
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args, GroupParams, Namespace
from gaussian_renderer import GaussianModel
from scene import Scene


def simply_setting_value(color_dc, color_rest):
    color_dc[:] = torch.tensor([[255, 0, 0]], dtype = torch.float32)
    color_rest[:] = torch.tensor([[255, 0, 0]], dtype = torch.float32)
    return color_dc, color_rest

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
        
        if self.transform != None:
            objidx = 132 # TODO: randomly choosing obj index (with some criteria)
            mask = scene.gaussians.object_mask == objidx
            scene.gaussians.transform_setup(mask, self.transform)

        return scene

    def __len__(self):
        return len(self.gs_scenes)