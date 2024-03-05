import cv2
from PIL import Image
import time
import torch
import numpy as np
import os
import pandas as pd
import time 
import datetime

import matplotlib.pyplot as plt

def get_time_str(start_time, curr_time):
    curr_time_str = datetime.datetime.fromtimestamp(curr_time).strftime('%Y-%m-%d %H:%M:%S')
    delta = datetime.timedelta(seconds=(curr_time - start_time))
    delta_str = str(delta - datetime.timedelta(microseconds=delta.microseconds))
    return curr_time_str, delta_str

def compute_depth_variance(img):
    flatten_values = img.flatten()
    return np.var(flatten_values)



filenames = []
labels = []

# root = '/home/elham/anti-spoofing/Evaluation/benchmarks/input_images'
root = 'DATASET_VALIDATION/real'
images = os.listdir(root)
images = [x for x in images if '.db' not in x]
for img in images:
    filenames.append(root +  "/" +img)
        

model_type = "DPT_Large"     # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
#model_type = "DPT_Hybrid"   # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
#model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)



midas = torch.hub.load("intel-isl/MiDaS", model_type)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas.to(device)
midas.eval()

midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
var_depth = []

if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
    transform = midas_transforms.dpt_transform
else:
    transform = midas_transforms.small_transform

start = time.time()
for i in range(len(filenames)):
    filename = filenames[i]
    img = cv2.imread(filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    input_batch = transform(img).to(device)

    with torch.no_grad():
        prediction = midas(input_batch)

        array = prediction.numpy()
        scaled_array = (array - array.min()) / (array.max() - array.min()) * 255
        path = "depth/" + os.path.split(filename)[0]
        np.savetxt(path + '/{}.txt'.format(os.path.split(filename)[1][:-4]), np.squeeze(scaled_array), fmt='%d')

        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

        # plt.figure()
        # array = prediction.numpy()
        # scaled_array = (array - array.min()) / (array.max() - array.min()) * 255
        # plt.imshow(scaled_array)
        # plt.savefig("depth/" + filename)
        # plt.show(block=True)
        # plt.close()




    output = prediction.cpu().numpy()

    # cv2.imwrite(label+"_out.jpg", output)
    # print(label + " depth variance : " + str(compute_depth_variance(output)))
    var_depth.append(compute_depth_variance(output))

end = time.time()
print('Elapsed time (s)=', end-start)

curr_time_str, elapsed_str = get_time_str(start, end)
print("Elapsed time: ", elapsed_str)

df = pd.DataFrame({'image':filenames, 'depth':var_depth})
df.to_csv('results_benchmarks_original_faces.csv', index = False)

#plt.imshow(output)