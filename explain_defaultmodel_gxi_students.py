import time
import os
import PIL.Image
import numpy as np

import torch

from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader

from typing import Iterator, Tuple, List


from getimagenetclasses import get_classes

from heatmaphelpers import imshow2

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


def get_gradxinp(model, image, device, clstoexplain):

    # make sure we work on a single input    
    if len(image.shape) != 4:
      print('wrong shape')
      exit()
    if image.shape[0]!=1:
      print('batchsize is not 1, but instead:', image.shape[0]) 
      exit() 

    model.eval() # !!!!!
 
 
 
    #with torch.no_grad(): #DO record graph of comp
    inputs = image.to(device)    
    
    # !!!
    inputs.requires_grad = True
    
    # prediction as before    
    outputs = model(inputs) 
      
    if clstoexplain <0:
      # get the max scoring one     
      clsindex = torch.argmax(outputs,1)
    else:
      clsindex = clstoexplain
      
    outputs[0,clsindex].backward() #compute the gradient

    print('grad.shape',inputs.grad.shape)
    
    #compute grad x input
    with torch.no_grad():
      explain = inputs.grad.data * inputs.data

    return outputs,explain

def runstuff_predictonly0(imgindex):
  #uses vanilla imagenet model
  from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
  
  #someparameters  
  batchsize_test=1 #batchsize - we work on a single image now
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

  #iterate over all image files, concatenate results

  for i,dic in enumerate(dataloader): 
  
    if i< imgindex:
      continue
    if i> imgindex:
      break  
  
    images = dic['image'] #dataloader returns a dict! 

    #predictions0 = predict0(model,  images , device)
    preds, explain= get_gradxinp(model, images, device, clstoexplain=-1)

    out= torch.topk(preds[0,:],k=3,dim=0)
    indices = out[1]
  
    print( 'for image index', i , 'top-3 predicted classes:'  )
    print( [ cls_list[k.item()] for k in indices ] )    

    imshow2(explain.to('cpu'),imgtensor = images.to('cpu'))

if __name__=='__main__':

  runstuff_predictonly0(imgindex = 3 )


