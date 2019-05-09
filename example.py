from skimage.io import imread
import os
import numpy as np
import matplotlib.pyplot as plt
import time
import torch
import torch.nn as nn
import sys
import matplotlib.pyplot as plt
import Cost

imgL_path = "data/left.png"
imgR_path = "data/right.png"
inputL_img = imread(imgL_path).astype(np.float32)
inputR_img = imread(imgR_path).astype(np.float32)

D = 192
imgL = torch.FloatTensor(inputL_img).cuda()
imgR = torch.FloatTensor(inputR_img).cuda()
imgL = torch.mean(imgL,2,keepdim = True)
imgR = torch.mean(imgR,2,keepdim = True)
cost = torch.zeros_like(imgL)
cost = torch.unsqueeze(cost,2)
cost = cost.repeat(1,1,D,1)
print(cost.shape)

torch.cuda.synchronize()
start_time = time.time()
Cost.Intrinsic_Cost(imgL,imgR,cost,7)
torch.cuda.synchronize()
deltatime = time.time() - start_time
print("SAD Time: " + str(deltatime))

cost = torch.squeeze(cost,3)
depth = torch.argmin(cost,2)
depth = depth.cpu().numpy()
plt.imsave("data/depth.png",depth,cmap = 'plasma', vmin = 0, vmax = D)
