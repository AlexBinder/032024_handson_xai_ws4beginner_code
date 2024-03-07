import time
import os
import PIL.Image
import numpy as np

import torch
import torch.nn as nn

from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader

from typing import Iterator, Tuple, List

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


def loadmodel(numcl,savedclassifier ,device):

  model = models.efficientnet_b0()
  
  #the next is efficientnet specific change of the last output layer, 
  #can iterate over model.named_modules() to see what the model contains
  num_ftrs = model.classifier[1].in_features
  model.classifier[1] = nn.Linear(num_ftrs, numcl) #reset the number of classes
  
  #load weights
  weights = torch.load(savedclassifier, map_location = device ) 
  model.load_state_dict( weights )
  
  return model

def runstuff_predictonly():

  #someparameters
  
  #uses custom imagenet model with custom number of classes
  savedclassifier = './flowers102model.pth' #model
  numcl=102 #number of classes
  
  batchsize_test=3 #batchsize
  device=torch.device('cpu') #device

  #custom model
  model = loadmodel(numcl,savedclassifier,device)
  model = model.to(device)
  
  #custom transforms
  data_transforms = {}
  customtransforms=transforms.Compose([
          transforms.Resize(256),
          transforms.CenterCrop(224),
          transforms.ToTensor(),
          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
      ])
  
  #get the data
  filelist = getfilelist_someimg()
  
  #dataset and dataloader
  dataset= dataset_filelist_nolabels(filelist  , transform= customtransforms)
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
  
    print( 'for image index', i , 'top-3 predicted classes:' , indices )


  return allpredictions

if __name__=='__main__':

  runstuff_predictonly()


