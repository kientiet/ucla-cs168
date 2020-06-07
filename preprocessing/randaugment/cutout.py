import random
import numpy as np

def cutout(mask_size, p, cutout_inside, mask_color=(0, 0, 0)):
  mask_size_half = mask_size // 2
  offset = 1 if mask_size % 2 == 0 else 0

  def _cutout(image):
    image = np.asarray(image).copy()

    if np.random.random() > p:
        return image

    h, w = image.shape[:2]

    if cutout_inside:
        cxmin, cxmax = mask_size_half, w + offset - mask_size_half
        cymin, cymax = mask_size_half, h + offset - mask_size_half
    else:
        cxmin, cxmax = 0, w + offset
        cymin, cymax = 0, h + offset

    cx = random.randint(cxmin, cxmax)
    cy = random.randint(cymin, cymax)
    xmin = cx - mask_size_half
    ymin = cy - mask_size_half
    xmax = xmin + mask_size
    ymax = ymin + mask_size
    xmin = max(0, xmin)
    ymin = max(0, ymin)
    xmax = min(w, xmax)
    ymax = min(h, ymax)
    image[ymin:ymax, xmin:xmax] = mask_color
    return image

  return _cutout
