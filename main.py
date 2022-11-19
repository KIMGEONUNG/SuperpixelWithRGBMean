from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import io
import skimage.measure
import matplotlib.pyplot as plt
import numpy as np


def dominant_grid(segments, winsize=16):
  """
  Write documents
  """
  num_seg = segments.max() + 1
  one_hot = np.eye(num_seg)[segments]
  sum_pooling = skimage.measure.block_reduce(one_hot, (winsize, winsize, 1),
                                             func=np.sum)
  idx_grid = np.argmax(sum_pooling, axis=-1)
  return idx_grid


def gen_sp2rgbmean(image, segments):
  """
  Generate a mapping from superpixel index to corresponding RGB mean value
  """

  # Define required values
  num_seg = segments.max() + 1
  dim_spatial = image.shape[:-1]  # (w, h)
  axis_spatial = tuple(range(len(dim_spatial)))
  dim_spectral = 3  # RGB

  # Convert label to one-hot encoding
  one_hot = np.eye(num_seg)[segments]
  one_hot_sum = one_hot.sum(axis_spatial)

  # Calculate mean RGB value according to the label
  pix_lab_rgb = one_hot.reshape(*dim_spatial, num_seg, 1) * image.reshape(
      *dim_spatial, 1, dim_spectral)
  sp2rgb = pix_lab_rgb.sum(axis_spatial)
  one_hot_sum_reshape = one_hot_sum.reshape(num_seg, 1).repeat(dim_spectral, 1)
  sp2rgb = sp2rgb / one_hot_sum_reshape

  return sp2rgb


def gen_mean_superpixel(image, num_seg=100, sigma=5):
  """
  SLIC
  """
  image_flat = image.reshape(-1, 3)
  w, h, _ = image.shape
  dim_spatial = w * h

  segments = slic(image, n_segments=num_seg, sigma=sigma)

  # Redefine the number of superpixel after the SLIC estimation
  num_seg = segments.max() + 1

  # Flatten spatial dimension to improve visibility
  segments_flat = segments.reshape(-1)

  # Convert label to one-hot encoding
  one_hot = np.eye(num_seg)[segments_flat]
  one_hot_sum = one_hot.sum(0)

  # Calculate mean RGB value according to the label
  pix_lab_rgb = one_hot.reshape(dim_spatial, num_seg, 1) * image_flat.reshape(
      dim_spatial, 1, 3)
  sp2rgb = pix_lab_rgb.sum(0)
  one_hot_sum_reshape = one_hot_sum.reshape(num_seg, 1).repeat(3, 1)
  sp2rgb = sp2rgb / one_hot_sum_reshape

  # Replace all RGB value with mean RGB value
  image_meansp = one_hot[..., None] * sp2rgb
  image_meansp = image_meansp.sum(-2)
  image_meansp = image_meansp.reshape(w, h, 3)

  return image_meansp, segments, sp2rgb


def main():
  image = img_as_float(io.imread('sample01_resize_256_256.jpg'))

  num_seg = 100
  image_meansp, segments, sp2rgb = gen_mean_superpixel(image, num_seg=num_seg)
  grid = dominant_grid(segments, num_seg=num_seg)
  dominant_color = sp2rgb[grid]

  fig = plt.figure("Superpixels -- %d segments" % (num_seg))
  ax = fig.add_subplot(1, 2, 1)
  ax.set_title("Superpixel with mean color")
  ax.imshow(mark_boundaries(image, segments))
  ax.imshow(image_meansp)
  plt.axis("off")

  ax = fig.add_subplot(1, 2, 2)
  ax.set_title("dominant_color")
  ax.imshow(dominant_color)
  plt.axis("off")

  plt.tight_layout()
  plt.show()


def main2():
  image = img_as_float(io.imread('sample01_resize_256_256.jpg'))
  num_seg = 100
  sigma = 5
  winsize = 16

  # What we exactly needs ====================================================
  segments = slic(image, n_segments=num_seg, sigma=sigma)
  sp2rgbmean = gen_sp2rgbmean(image, segments)  # The most important data
  idx_grid = dominant_grid(segments, winsize=winsize)
  dominant_color = sp2rgbmean[idx_grid]
  # print(dominant_color.shape)  # (w/16, h/16, 3)
  # ==========================================================================

  fig = plt.figure("hello")
  ax = fig.add_subplot(1, 1, 1)
  ax.set_title("dominant_color")
  ax.imshow(dominant_color)
  plt.axis("off")

  plt.tight_layout()
  plt.show()


if __name__ == "__main__":
  main2()
