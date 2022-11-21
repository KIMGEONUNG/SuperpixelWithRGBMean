from skimage.segmentation import slic
from os.path import join
from skimage.util import img_as_float
from skimage import io
import skimage.measure
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import torch
import torch.nn as nn


def parse():
  p = argparse.ArgumentParser()
  p.add_argument("--winsize", type=int, default=16)
  p.add_argument("--path_input", type=str, default="sample01.jpg")
  p.add_argument("--save", action='store_true')
  p.add_argument("--sigma", type=float, default=5.0)
  p.add_argument("--num_seg", type=int, default=100)
  return p.parse_args()


def dominant_pool_2d(spatial: np.ndarray, winsize=16):
  """
  Return a 2-D array with a pooling operation.
  The pooling operation is to select the most dominant value for each window.
  This assumes that the input 'spatial' has discrete values like index or lablel.
  To circumvent an use of iterative loop, we use a trick with one-hot encoding
  and 'skimage.measure.block_reduce' function.

  Parameters
  ----------
  spatial : int ndarray of shape (width, hight)
    The spatial is represented by int label, not one-hot encoding
  winsize : int, optional
    Length of sweeping window

  Returns
  -------
  pool : ndarray of shape (N,M)
    The pooling results.

  """
  num_seg = spatial.max() + 1
  one_hot = np.eye(num_seg)[spatial]
  sum_pooling = skimage.measure.block_reduce(one_hot, (winsize, winsize, 1),
                                             func=np.sum)
  pool = np.argmax(sum_pooling, axis=-1)
  return pool


def gen_sp2rgbmean(image: np.ndarray, segments: np.ndarray):
  """
  Generate a mapping from superpixel index to the corresponding RGB mean value

  Parameters
  ----------
  image : image ndarray of shape (width, hight, dim_spectrum)
    The spatial is represented by int label, not one-hot encoding
  segments : int ndarray of shape (width, hight)
    Segmentation labels for each pixel

  Returns
  -------
  sp2rgb : ndarray of shape (N, dim_spectrum)
    Mapping of superpixel index to RGB mean value.
    N is the number of segmentation label.
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


def main1():
  """
  Estimate dominant colors for each spatial region.
  """
  args = parse()
  image = img_as_float(io.imread(args.path_input))
  num_seg = args.num_seg
  sigma = args.sigma
  winsize = args.winsize

  # What we exactly needs ====================================================
  segments = slic(image, n_segments=num_seg, sigma=sigma)
  sp2rgbmean = gen_sp2rgbmean(image, segments)  # The most important data
  idx_grid = dominant_pool_2d(segments, winsize=winsize)
  dominant_color = sp2rgbmean[idx_grid]
  # print(dominant_color.shape)  # (w/16, h/16, 3)
  # ==========================================================================

  if args.save:
    io.imsave(join('outputs', args.path_input), dominant_color)
  else:
    fig = plt.figure("hello")
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title("dominant_color")
    ax.imshow(dominant_color)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def main2():
  """
  Estimate superpixels with mean color.
  """
  args = parse()
  image = img_as_float(io.imread(args.path_input))
  num_seg = args.num_seg
  sigma = args.sigma

  # What we exactly needs ====================================================
  segments = slic(image, n_segments=num_seg, sigma=sigma)
  sp2rgbmean = gen_sp2rgbmean(image, segments)  # The most important data
  superpixel = sp2rgbmean[segments]
  # print(dominant_color.shape)  # (w/16, h/16, 3)
  # ==========================================================================

  if args.save:
    io.imsave(join('outputs', args.path_input), superpixel)
  else:
    fig = plt.figure("hello")
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title("dominant_color")
    ax.imshow(superpixel)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def gen_mask_hint_label(
    dim_spatial=(16, 16),
    prop_mask_all=0.03,
    num_mask_from=16,
    num_mask_to=256,
    prop_hint=0.7,
    num_hint_from=1,
    num_hint_to=16,
):
  """
  The function sample masked and hint regions.
  Note that the hint points are sampled among masked regions.

  Parameters
  ----------
  dim_spatial : an tuple of shape (W, H)
  prop_mask_all : float for probability
  num_mask_from : int number for minimum value of the number of sampling mask
  num_mask_to : int number for upper bound of the number of sampling mask
  prop_hint : float for probability
  num_hint_from : int number for minimum value of the number of sampling hint
  num_hint_to : int number for minimum value of the number of sampling hint

  Returns
  -------
  mask : ndarray of shape (W, H)
    the mask has three types of value
     0 --> masked points
     1 --> unmasked points, i.e. preserve original values
    -1 --> the points using color hint
  """

  dim_spatial_flat = dim_spatial[0] * dim_spatial[1]

  # MASK SYNTHESIS
  mask = np.ones(dim_spatial_flat).astype('int')
  is_all_mask = np.random.binomial(1, p=prop_mask_all, size=1)[0]
  if is_all_mask:
    idx_mask = random.sample(range(0, dim_spatial_flat), dim_spatial_flat)
  else:
    num_mask = np.random.randint(num_mask_from, num_mask_to + 1)
    idx_mask = random.sample(range(0, dim_spatial_flat), num_mask)

  mask[idx_mask] = 0

  # HINT SELECTION
  is_hint = np.random.binomial(1, p=prop_hint, size=1)[0]
  if is_hint:
    num_hint = np.random.randint(num_hint_from, num_hint_to + 1)
    idx_hint = random.sample(idx_mask, num_hint)
    mask[idx_hint] = -1

  mask = mask.reshape(*dim_spatial)

  return mask


def main3():
  """
  Mask and hint sampling example
  - RGB image 
  - gray scale 
  """
  dim_embd = 1
  z_feat = torch.randn(3, dim_embd, 16, 16)
  mask = gen_mask_hint_label()

  # print(z_feat)
  # print(z_feat.shape)

  print(mask)
  print(mask <=0)
  z_feat[mask <= 0] = 0
  print(z_feat)

  exit()
  token_mask = torch.randn(dim_embd)

  # print(mask  0)
  exit()

  args = parse()
  image = img_as_float(io.imread(args.path_input))
  num_seg = args.num_seg
  sigma = args.sigma
  winsize = args.winsize

  # What we exactly needs ====================================================
  segments = slic(image, n_segments=num_seg, sigma=sigma)
  sp2rgbmean = gen_sp2rgbmean(image, segments)  # The most important data
  idx_grid = dominant_pool_2d(segments, winsize=winsize)
  dominant_color = sp2rgbmean[idx_grid]
  # print(dominant_color.shape)  # (w/16, h/16, 3)
  # ==========================================================================

  if args.save:
    io.imsave(join('outputs', args.path_input), dominant_color)
  else:
    fig = plt.figure("hello")
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title("dominant_color")
    ax.imshow(dominant_color)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def tmp_embeding():
  embd = nn.Embedding(num_embeddings=2, embedding_dim=3)
  x = torch.IntTensor([0, 1])
  y = embd(x)
  print(embd.weight)
  print(y)
  pass


if __name__ == "__main__":
  # main1()
  # main2()
  # tmp_embeding()
  main3()
