import os
import io

from skimage import  transform
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from torch.utils.data import Dataset, DataLoader

import numpy as np
from PIL import Image

from data_loader import RescaleT
from data_loader import ToTensorLab

from model.u2net import U2NET, U2NETP
# from model import U2NETP # small version u2net 4.7 MB
import base64
from io import BytesIO



def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)
    dn = (d - mi) / (ma - mi)
    return dn


def preprocess(image):
    label_3 = np.zeros(image.shape)
    label = np.zeros(label_3.shape[0:2])

    if (3 == len(label_3.shape)):
        label = label_3[:, :, 0]
    elif (2 == len(label_3.shape)):
        label = label_3

    if (3 == len(image.shape) and 2 == len(label.shape)):
        label = label[:, :, np.newaxis]
    elif (2 == len(image.shape) and 2 == len(label.shape)):
        image = image[:, :, np.newaxis]
        label = label[:, :, np.newaxis]

    transform = transforms.Compose([RescaleT(320), ToTensorLab(flag=0)])
    sample = transform({
        'imidx': np.array([0]),
        'image': image,
        'label': label
    })

    return sample


def initialize_net():
    currentDir = os.path.dirname(__file__)

    model_name = 'u2net'
    model_dir = os.path.join(currentDir, 'saved_models',
                         model_name, model_name + '.pth')

    print("Loading U-2-Net...")
    net = U2NET(3, 1)
    net.load_state_dict(torch.load(model_dir))
    if torch.cuda.is_available():
        net.cuda()
 
    return net

net = initialize_net()  # This will initialize the net when this script is imported

def run(img, original_size):
    torch.cuda.empty_cache()

    sample = preprocess(img)
    inputs_test = sample['image'].unsqueeze(0)
    inputs_test = inputs_test.type(torch.FloatTensor)

    if torch.cuda.is_available():
        inputs_test = Variable(inputs_test.cuda())
    else:
        inputs_test = Variable(inputs_test)

    d1, d2, d3, d4, d5, d6, d7 = net(inputs_test)

    # Normalization.
    pred = d1[:, 0, :, :]
    predict = normPRED(pred)

    # Convert to PIL Image
    predict = predict.squeeze()
    predict_np = predict.cpu().data.numpy()
    im = Image.fromarray(predict_np * 255).convert('RGB')

    # Resize the image to original dimensions
    im = im.resize(original_size, resample=Image.NEAREST)
    # Cleanup.
    del d1, d2, d3, d4, d5, d6, d7

    return im

def bgRemover(imgPath) :
    # Convert string data to PIL Image
   
    
    img = Image.open( BytesIO(
            base64.b64decode(imgPath )
        ))

     # Ensure image size is under 1024
    original_size = img.size  # store original size
   
    
    currentDir = os.path.dirname(__file__)

    model_name = 'u2net'
    model_dir = os.path.join(currentDir, 'saved_models',
                         model_name, model_name + '.pth')

    print("Loading U-2-Net...")
    print(model_dir)
    net = U2NET(3, 1)
    net.load_state_dict(torch.load(model_dir))
    if torch.cuda.is_available():
        net.cuda()
 

    # Process Image
    res = run(np.array(img), original_size)

    # Save to buffer
    buff = io.BytesIO()
    res.save(buff, 'PNG')
    buff.seek(0)

    image_base64 = base64.b64encode(buff.getvalue()).decode('utf-8')

    return image_base64
    # Save the buffer contents to a file
    # output_image_path = 'output_image.png'  # specify the output file name
    # with open(output_image_path, 'wb') as f:
    #     f.write(buff.getvalue())  # write the contents of the buffer to the file

    

 