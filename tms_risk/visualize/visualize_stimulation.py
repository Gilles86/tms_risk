import os
import os.path as op
import matplotlib.colors as mcolors

from tms_risk.utils.data import get_tms_subjects
from nilearn import surface

import cortex
import numpy as np
from matplotlib import colors, cm

subjects = [f'{subject:02d}' for subject in get_tms_subjects()]
subjects.pop(subjects.index('45'))
bids_folder = '/data/ds-tmsrisk'

ds = {}

ims = []

for subject in subjects:
    im = op.join(bids_folder, 'derivatives', 'ips_masks', f'sub-{subject}', 'anat', f'sub-{subject}_space-fsaverage_desc-NPCr5mm_geodesic_hemi-R.anat.gii')
    im = surface.load_surf_data(im) > 0.0

    ims.append(im)

    # vertex = np.concatenate((np.zeros_like(im), im))
    # ds['sub-' + subject] = cortex.Vertex(vertex, 'fsaverage', vmin=0.0, vmax=1.0, cmap='viridis')


summed_masks = np.sum(ims, 0)
summed_masks = np.concatenate((np.zeros_like(summed_masks), summed_masks))

ds['mean'] = cortex.Vertex(summed_masks, 'fsaverage', vmin=0.0, vmax=5.0)

data  = summed_masks.copy()

vmin = 0.0
vmax = summed_masks.max()
print('Vmax: ', vmax)

data = np.clip((data - vmin) / (vmax - vmin), 0., .99)

cmap = 'coolwarm'
data[data < 0.1] = 0.0
alpha = data > 0.1
data[alpha < 0.01] = 0

red, green, blue = getattr(cm, cmap)(data,)[:, :3].T

    # Get curvature
curv = cortex.db.get_surfinfo('fsaverage')
# Adjust curvature contrast / color. Alternately, you could work
#     # with curv.data, maybe threshold it, and apply a color map.
curv.data = np.sign(curv.data.data) * .25
curv.vmin = -1
curv.vmax = 1
curv.cmap = 'gray'
curv_rgb = np.vstack([curv.raw.red.data, curv.raw.green.data, curv.raw.blue.data]).astype(np.float32)

vx_rgb = (np.vstack([red.data, green.data, blue.data]) * 255.).astype(np.float32)

display_data = vx_rgb * alpha[np.newaxis, :] + curv_rgb * (1.-alpha[np.newaxis, :])

ds['mean_thr'] = cortex.VertexRGB(*display_data.astype(np.uint8), 'fsaverage')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def generate_bitmaps(n, length):
    """ Generate random 1D bitmaps for demonstration purposes. """
    return np.random.rand(n, length) > 0.5

def mix_colors(bitmaps):
    """ Mix colors based on overlapping regions in 1D bitmaps using complex numbers for hues. """
    n, length = bitmaps.shape
    hues = np.linspace(0, 360, n, endpoint=False)  # Unique hue for each subject in degrees
    
    # Prepare complex accumulators for hues and counts for saturation
    hue_real = np.zeros(length)
    hue_imag = np.zeros(length)
    overlaps = np.zeros(length)
    
    # Sum up complex representations of hues
    for i, bitmap in enumerate(bitmaps):
        angle = np.deg2rad(hues[i])  # Convert degrees to radians
        mask = bitmap > 0
        hue_real += mask * np.cos(angle)
        hue_imag += mask * np.sin(angle)
        overlaps += mask
    
    # Calculate the average hue from the sum of complex numbers
    avg_hue = np.mod(np.rad2deg(np.arctan2(hue_imag, hue_real)), 360) / 360  # Normalize to [0, 1]
    valid = overlaps > 0  # Avoid division by zero
    
    # Adjust saturation based on overlap, using a steeper decay function
    saturation = np.where(valid, 1 / np.power(overlaps, 1.5), 0)
    
    # Adjust value based on saturation: higher for low saturation, lower for high saturation
    value = np.where(valid, 1 - saturation * 0.25, 1)
    
    # Create the final HSV image (1D)
    image_hsv = np.zeros((length, 3))  # Initialize HSV image
    image_hsv[:, 0] = avg_hue  # Hue
    image_hsv[:, 1] = saturation  # Saturation
    image_hsv[:, 2] = value  # Value (brightness)
    
    # Convert HSV to RGB for display
    image_rgb = mcolors.hsv_to_rgb(image_hsv)
    return image_rgb * 255.


display_data = mix_colors(np.array(ims))
display_data = np.concatenate((np.zeros_like(display_data), display_data)).T

display_data = display_data * alpha[np.newaxis, :] + curv_rgb * (1.-alpha[np.newaxis, :])
ds['bla'] = cortex.VertexRGB(*display_data.astype(np.uint8), 'fsaverage')

cortex.webshow(ds)