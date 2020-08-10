import torch


def data_to_torch(data, torch_type, device):
    data = torch.tensor(data, device=device, dtype=torch_type)
    return data
