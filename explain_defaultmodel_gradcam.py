import time
import os
import PIL.Image
import numpy as np

import torch

from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader

from typing import Iterator, Tuple, List


from torchvision.models import resnet50

from getimagenetclasses import get_classes
from heatmaphelpers import *

from pytorch_grad_cam import GradCAM #, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image






# helper routine for the filenames on disk
def getfilelist_someimg():
  ls=[]
  ls.append('./somepascalvoc/2007_001430.jpg')
  ls.append('./somepascalvoc/2007_001678.jpg')
  ls.append('./somepascalvoc/2007_001594.jpg')
  ls.append('./somepascalvoc/2007_001733.jpg')
  ls.append('./somepascalvoc/2007_001763.jpg')
  
  return ls

###################################################
#custom dataset
###################################################
class dataset_filelist_nolabels(Dataset):
  def __init__(self, imgfilenames, transform=None):

    self.transform = transform
    self.imgfilenames=imgfilenames

  def __len__(self):
      return len(self.imgfilenames)

  def __getitem__(self, idx):

    image = PIL.Image.open(self.imgfilenames[idx]).convert('RGB')
    if self.transform:
      image = self.transform(image) # resizing,crop,normalization here

    sample = {'image': image,  'filename': self.imgfilenames[idx]}

    return sample
    

def run(imgindex):

  model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
  model.eval() 

  target_layers = [model.layer4[-3]]


  transforms = models.ResNet50_Weights.DEFAULT.transforms()
  
  #get the data
  filelist = getfilelist_someimg()
  
  #get the possible outputclasses
  
  cls_list= get_classes()  
  
  #dataset and dataloader
  batchsize_test =1
  dataset= dataset_filelist_nolabels(filelist  , transform= transforms)
  dataloader = torch.utils.data.DataLoader(dataset, batchsize_test, shuffle=False)  #a bit overkill, could just batch them manually

  #iterate over all image files, concatenate results

  for i,dic in enumerate(dataloader): 
  
    if i< imgindex:
      continue
    if i> imgindex:
      break  
  
    images = dic['image'] #dataloader returns a dict!


    input_tensor = images # Create an input tensor image for your model..
    # Note: input_tensor can be a batch tensor with several images!

    # Construct the CAM object once, and then re-use it on many images:
    cam = GradCAM(model=model, target_layers=target_layers)

    # You can also use it within a with statement, to make sure it is freed,
    # In case you need to re-create it inside an outer loop:
    # with GradCAM(model=model, target_layers=target_layers) as cam:
    #   ...

    # We have to specify the target we want to generate
    # the Class Activation Maps for.
    # If targets is None, the highest scoring category
    # will be used for every image in the batch.
    # Here we use ClassifierOutputTarget, but you can define your own custom targets
    # That are, for example, combinations of categories, or specific outputs in a non standard model.

    targets = None

    # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)

    # In this example grayscale_cam has only one image in the batch:
    grayscale_cam = grayscale_cam[0, :]
    print(grayscale_cam.shape)
    
    #visualization = show_cam_on_image( images[0,:], grayscale_cam, use_rgb=True)
    imshow2(torch.tile(torch.from_numpy(grayscale_cam).to('cpu').unsqueeze(0).unsqueeze(0), [1,2,1,1]),imgtensor = images.to('cpu'))
    
    # You can also get the model outputs without having to re-inference
    model_outputs = cam.outputs
    
if __name__=='__main__':

    run(imgindex = 3 )   
