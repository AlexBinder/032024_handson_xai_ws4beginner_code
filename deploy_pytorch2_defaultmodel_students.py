import time
import os
import PIL.Image
import numpy as np

import torch

from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader

from typing import Iterator, Tuple, List


from getimagenetclasses import get_classes

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
    


###################################################
#predict code starts here
##################################################
def predict0(model, setofimages, device):

    model.eval() # !!!!!
    
    with torch.no_grad(): #dont record graph of comp
           
      inputs = setofimages.to(device)        
      outputs = model(inputs) 
      out= outputs.to('cpu') #back to cpu
          
    return out

def runstuff_predictonly0():
  #uses vanilla imagenet model
  from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
  
  #someparameters  
  batchsize_test=2 #batchsize
  device=torch.device('cpu') #device

  #model
  model = efficientnet_b0(weights= EfficientNet_B0_Weights.DEFAULT)
  model = model.to(device)
  
  #transforms
  transforms = EfficientNet_B0_Weights.DEFAULT.transforms()
  
  #get the data
  filelist = getfilelist_someimg()
  
  #get the possible outputclasses
  
  cls_list= get_classes()  
  
  #dataset and dataloader
  dataset= dataset_filelist_nolabels(filelist  , transform= transforms)
  dataloader = torch.utils.data.DataLoader(dataset, batchsize_test, shuffle=False)  #a bit overkill, could just batch them manually
  
  # for one image:
  #images = next(iter( dataloader ))['image'] #returns a dict! 
  #predictions=predict0(model,  images , device)

  #iterate over all image files, concatenate results
  allpredictions = None
  for dic in dataloader: 
  
    images = dic['image'] #dataloader returns a dict! 
    predictions0 = predict0(model,  images , device)
    
    #concat all predictions
    print('single pred shape', predictions0.shape)
    if allpredictions is None:
      allpredictions = predictions0 #comes already with batch dimension
    else: 
      allpredictions = torch.cat((allpredictions,predictions0),dim=0)
  
  print('total pred shape', allpredictions.shape)
  
  

  
  
  
  for i in range(allpredictions.shape[0]):
  
    out= torch.topk(allpredictions[i,:],k=3,dim=0)
    indices = out[1]
  
    print( 'for image index', i , 'top-3 predicted classes:'  )
    print( [ cls_list[k.item()] for k in indices ] )
    
  
  return allpredictions



if __name__=='__main__':

  runstuff_predictonly0()


