import numpy as np
import torch
import torch.nn.functional as F


def get_event_tensor_padded(event_path, padding_bottom):
    padding = (0, 0, 0, padding_bottom)
    ev_data = np.load(event_path)
    t_ev_data = torch.from_numpy(ev_data)
    t_ev_data = F.pad(t_ev_data, padding, mode="constant", value=0)
    t_ev_data = t_ev_data.float()
    return t_ev_data


def get_rgb_tensor_padded(rgb_path, padding_bottom):
    padding = (0, 0, 0, padding_bottom)
    rgb_data = np.load(rgb_path)
    t_rgb_data = torch.from_numpy(rgb_data).permute(2, 0, 1)
    t_rgb_data = F.pad(t_rgb_data, padding, mode="constant", value=0)
    t_rgb_data = t_rgb_data.float() / 255
    return t_rgb_data
