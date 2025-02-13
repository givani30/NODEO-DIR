import numpy as np
import nibabel as nib
from Loss import *
import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer
    """

    def __init__(self, size, mode='bilinear'):
        super().__init__()

        self.mode = mode

        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)

        # registering the grid as a buffer cleanly moves it to the GPU, but it also
        # adds it to the state dict. this is annoying since everything in the state dict
        # is included when saving weights to disk, so the model files are way bigger
        # than they need to be. so far, there does not appear to be an elegant solution.
        # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
        self.register_buffer('grid', grid)

    def forward(self, src, flow, return_phi=False):
        # new locations
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        # need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # move channels dim to last position
        # also not sure why, but the channels need to be reversed
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        if return_phi:
            return F.grid_sample(src, new_locs, align_corners=True, mode=self.mode), new_locs
        else:
            return F.grid_sample(src, new_locs, align_corners=True, mode=self.mode)

def load_nii(path):
    X = nib.load(path)
    X = X.get_fdata()
    X=np.squeeze(X)
    print(" Break")
    return X

def save_nii(img, savename):
    affine = np.diag([1, 1, 1,1])
    new_img = nib.nifti1.Nifti1Image(img, affine, header=None)
    nib.save(new_img, savename)

def generate_grid3D_tensor(shape):
    x_grid = torch.linspace(-1., 1., shape[0])
    y_grid = torch.linspace(-1., 1., shape[1])
    x_grid, y_grid = torch.meshgrid(x_grid, y_grid)

    # Note that default the dimension in the grid is reversed:
    # z, y, x
    grid = torch.stack([ y_grid, x_grid], dim=0)
    return grid

def dice(array1, array2, labels):
    """
    Computes the dice overlap between two arrays for a given set of integer labels.
    """
    dicem = np.zeros(len(labels))
    for idx, label in enumerate(labels):
        top = 2 * np.sum(np.logical_and(array1 == label, array2 == label))
        bottom = np.sum(array1 == label) + np.sum(array2 == label)
        bottom = np.maximum(bottom, np.finfo(float).eps)  # add epsilon
        dicem[idx] = top / bottom
    return dicem


def plotDeformationField2D(flow, save_path=None):
    """
    Plots the deformation field as a vector field.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.colors import LinearSegmentedColormap

    # create color map
    flow=flow.cpu().numpy()
    flow=flow.reshape()
    colors = [(0, 0, 0), (0, 0, 1), (0, 1, 0), (1, 0, 0), (1, 1, 0), (1, 0, 1), (0, 1, 1), (1, 1, 1)]
    cmap_name = 'my_list'
    cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=8)

    # plot
    plt.figure()
    plt.imshow(flow[0, 0, ...], cmap=cm)
    plt.quiver(flow[0, 1, ...], flow[0, 0, ...], scale=10, color='w')
    plt.axis('off')
    plt.title('Deformation Field')
    red_patch = mpatches.Patch(color='red', label='Deformation Field')
    plt.legend(handles=[red_patch])
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()
