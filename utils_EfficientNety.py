# from preconditions import preconditions
# from matplotlib import pyplot as plt
# import matplotlib.cm
import torch
# import cv2
import numpy as np
from scipy import fftpack
import collections
import os
import torchvision.transforms as transforms
from scipy.ndimage import gaussian_filter


def euclid_dist(t1, t2):
    return np.sqrt(((t1 - t2) ** 2).sum(axis=0))


#-------------- this function only apply the fft (the input is image and the output is fft magnitude in size ...)
def fft_fun(image):
    _, _, M, N = image.shape
    image_low_pass_filter = gaussian_filter(image, sigma=1)
    image_high_pass_filter = image - image_low_pass_filter
    fft_components = torch.fft.fft(image)
    # fft_components = fftpack.fftn(image)
    fft_components = fft_components.cpu().numpy()
    fft_magnitude = np.abs(fft_components)
    # fft_magnitude = torch.fft.fftshift(fft_magnitude)
    fft_magnitude = fftpack.fftshift(fft_magnitude)
    K = 25
    H = M
    W = N
    # print("---------------", len(fft_magnitude))
    for matt in fft_magnitude:
        # matt[M // 2 - K: M // 2 + K, N // 2 - K: N // 2 + K] = 0
    # print('fft_magnitude', fft_magnitude.shape)
        matt2 = matt.copy()
        matt2[0 : H // 2 - K, 0: W // 2 - K] = 0
        matt2[H // 2 + K : H, W // 2 + K : W] = 0 
        matt2[0 : H // 2 - K , W // 2 + K: W] = 0
        matt2[H // 2 + K : H, 0:W //2 - K] = 0
        matt = matt - matt2
    
    diff_matt = []
    for matt in fft_magnitude:
        # print('------------ matt', matt.shape)
        matt = np.squeeze(np.array(matt))
        mat = np.abs(matt[0:M // 2 - K, 0:N // 2 - K] - matt[0:M // 2 - K, N // 2 + K:N])
        # print('------------ mat', mat.shape)
        diff_matt.append(mat)
    #print('diff_matt', np.shape(diff_matt), np.shape(diff_matt[0]), len(diff_matt))

        
    return torch.tensor(diff_matt)


def fft_hash_fun(image, no_peaks):
    _, _, M, N = image.shape
    image_low_pass_filter = gaussian_filter(image, sigma=1)
    image_high_pass_filter = image - image_low_pass_filter
    fft_components = torch.fft.fft(image)
    # fft_components = fftpack.fftn(image)
    fft_components = fft_components.cpu().numpy()
    fft_magnitude = np.abs(fft_components)
    # fft_magnitude = torch.fft.fftshift(fft_magnitude)
    fft_magnitude = fftpack.fftshift(fft_magnitude)
    K = 5
    # print("---------------", len(fft_magnitude))
    for matt in fft_magnitude:
        matt[M // 2 - K: M // 2 + K, N // 2 - K: N // 2 + K] = 0
    # print('fft_magnitude', fft_magnitude.shape)
    
    diff_matt = []
    for matt in fft_magnitude:
        # print('------------ matt', matt.shape)
        matt = np.squeeze(np.array(matt))
        mat = np.abs(matt[0:M // 2 - K, 0:N // 2 - K] - matt[0:M // 2 - K, N // 2 + K:N])
        # print('------------ mat', mat.shape)
        diff_matt.append(mat)
    # print('diff_matt', np.shape(diff_matt), np.shape(diff_matt[0]), len(diff_matt))

    peak_cand = []
    for matt in diff_matt:
        # sorts the unique elements from the nested list matt in ascending order based on their counts and returns the sorted keys.
        peaks_dif = sorted(collections.Counter(x for xs in matt for x in set(xs)).keys())
        peak_cand.append(peaks_dif[-no_peaks:])
    # print('peak_cand', peak_cand)

    row_list = []
    col_list = []
    distance_list = []
    for idx, matt1 in enumerate(diff_matt):
        row = []
        col = []
        distance = []
        for cand in range(no_peaks - 1, -1, -1):
            for i, lst in enumerate(matt1):
                # flat_list = [item for sublist in dif for item in sublist]
                for j, element in enumerate(lst):
                    if element == peak_cand[idx][cand]:
                        row.append(i)
                        col.append(j)
                        # distance.append(euclid_dist(np.array([i, j]), np.array([114, 114])))
                        break # to avoid select one value several times.
        row_list.append(row)
        col_list.append(col)
        # distance_list.append(distance)
    # print('row', row_list)
    # print('col', col_list) 
    h_code_list = []
    for idx in range(len(row_list)):
        row = row_list[idx]
        col = col_list[idx]
        min_r = min(row)
        min_c = min(col)
        max_r = max(row)
        max_c = max(col)
        h_code = []        
        for i in range(len(row)):
            dist_A = euclid_dist(np.array([row[i], col[i]]), np.array([max_r + 1, min_c - 1]))
            dist_B = euclid_dist(np.array([row[i], col[i]]), np.array([min_r - 1, max_c + 1]))
            if dist_A < dist_B:
                h_code.append(0)
            else:
                h_code.append(1)
        h_code_list.append(h_code)
    # print('h_code_list', h_code_list)

    with open('peaks' + str(no_peaks) + '_Musab_ID.txt', 'a') as f:
        f.write("%s\t" % 'row')
        for item in row:
            # item = "{:.2f}".format(item[0])
            f.write("%s\t\t" % item)
        f.write("\n")
        f.write("%s\t" % 'col')
        for item in col:
            # item = "{:.2f}".format(item[0])
            f.write("%s\t\t" % item)
        f.write("\n")
        # f.write("%s\t %.2f\t" % ('Pval', peak_cand[no_peaks - 1]))
        for ind in range(no_peaks - 2, -1):
            f.write("%.2f\t\t" % peak_cand[ind])

        f.write("avg = %.2f\t\t" % np.average(peak_cand))
        f.write("\n")
        f.write("%s\t" % 'dist')
        for item in distance:
            f.write("%.2f\t\t" % item)
        f.write("\n")
        f.write("\n")
    return torch.tensor(h_code_list)


def simple_save_images(nn_noisy_image, name):
    nn_noisy_image = nn_noisy_image.cpu()[1, :, :, :]
    nn_noisy_image_numpy = nn_noisy_image.detach().numpy()
    norm_noisy_generated = cv2.normalize(nn_noisy_image_numpy, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
                                         dtype=cv2.CV_32F)

    norm_noisy_generated = norm_noisy_generated.astype(np.uint8)
    norm_noisy_generated = np.swapaxes(norm_noisy_generated, 0, 2)
    norm_noisy_generated = np.swapaxes(norm_noisy_generated, 0, 1)
    cv2.imwrite(name, norm_noisy_generated)


def DepthNorm(depth, max_depth=1000.0):
    return max_depth / depth


class AverageMeter(object):
    def __init__(self):
        self.count = None
        self.avg = None
        self.val = None
        self.sum = None
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def colorize(value, vmin=10, vmax=1000, cmap='plasma'):
    value = value.cpu().numpy()[0, :, :]

    # normalize
    vmin = value.min() if vmin is None else vmin
    vmax = value.max() if vmax is None else vmax
    if vmin != vmax:
        value = (value - vmin) / (vmax - vmin)  # vmin..vmax
    else:
        # Avoid 0-division
        value = value * 0.
    # squeeze last dim if it exists
    # value = value.squeeze(axis=0)

    cmapper = matplotlib.cm.get_cmap(cmap)
    value = cmapper(value, bytes=True)  # (nxmx4)

    img = value[:, :, :3]

    return img.transpose((2, 0, 1))


# custom weights initialization called on gen and disc model
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias) 





