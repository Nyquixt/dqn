import numpy as np 

def to_torch_image(obs):
    # convert from HWC to CHW for torch, in this case 84x84x4 to 4x84x84
    obs = np.array(obs)
    return np.moveaxis(obs, 2, 0)

def to_torch_image_batch(obs):
    # convert from HWC to CHW for torch, in this case 84x84x4 to 4x84x84
    obs = np.array(obs)
    return np.moveaxis(obs, 3, 1)