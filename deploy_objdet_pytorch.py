
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights

from torchvision.models.detection import fcos_resnet50_fpn, FCOS_ResNet50_FPN_Weights

from torchvision.models.detection import ssdlite320_mobilenet_v3_large, SSDLite320_MobileNet_V3_Large_Weights
from torchvision.io.image import read_image
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image

import torch

from typing import Iterator, Tuple, List



def runstuff_objdet_singlefile(fn):

  # load weights
  weights = SSDLite320_MobileNet_V3_Large_Weights.COCO_V1

  #get preprocessors
  preprocess = weights.transforms()
  

  # create model from weightd
  model = ssdlite320_mobilenet_v3_large(weights=weights, detections_per_img=10)#score_thresh=0.1) #detections_per_img=10)
  
  # eval mode  
  model.eval()
  
  # step 2: read the image
  print('fn',fn)
  img = read_image(fn)
  print(type(img))
  batch = [preprocess(img)]
  print(type(batch[0]))

  # Step 3: apply the model 
  with torch.no_grad():
    prediction = model(batch)[0]
  
  print('preds', prediction["boxes"].shape, prediction["labels"].shape)
  print('first box coords',prediction["boxes"][0,:])
  print('all predicted labels', prediction["labels"][:])


  # step 4 visualize the prediction
  if 0==0: # way to switch bbox drawing on or off 
  
    labels = [weights.meta["categories"][i] for i in prediction["labels"]]
    box = draw_bounding_boxes(img, boxes=prediction["boxes"],
                              labels=labels,
                              colors="red",
                              width=4, font_size=30)
    im = to_pil_image(box.detach())
    im.show()  
  return (prediction["boxes"].cpu().tolist(),prediction["labels"].cpu().tolist() )


##### if input is a filelist

def runstuff_objdet_filelist(fnames: List[str]):

  if len(fnames)==0:
    return (None,None)

  nummaxdet = 10 #8000 # difference to before, decides how much to pad!

  weights = SSDLite320_MobileNet_V3_Large_Weights.COCO_V1
  model = ssdlite320_mobilenet_v3_large(weights=weights, detections_per_img=nummaxdet)
  model.eval()
  
  preprocess = weights.transforms()
  
  for i,fn in enumerate(fnames): 
  
    print('fn',fn)
    img = read_image(fn)
    print(type(img))
    batch = [preprocess(img)]
    print(type(batch[0]))

    # Step 4: Use the model and visualize the prediction
    with torch.no_grad():
      prediction = model(batch)[0]
    
    print('preds', prediction["boxes"].shape, prediction["labels"].shape)
    #print('first box coords',prediction["boxes"][0,:])
    #print('all predicted labels', prediction["labels"][:])
  
    #pad to nummaxdet if needed
    if prediction["boxes"].shape[0]< nummaxdet:
      padsize = nummaxdet-prediction["boxes"].shape[0]
      print('padsize at',i,'=',padsize)
      boxes = torch.cat( (prediction["boxes"], torch.full(size=(padsize,4), fill_value=-1.0, dtype = prediction["boxes"].dtype ) ), dim = 0 )
      cls = torch.cat( (prediction["labels"], torch.full(size=(padsize,), fill_value=-1.0, dtype = prediction["labels"].dtype ) ),dim=0)
    else: #just reference
      boxes = prediction["boxes"]
      cls = prediction["labels"]
  
    #concat
    if i==0:
      allboxes = boxes.unsqueeze(0)
      allcls = cls.unsqueeze(0)
    else:
      allboxes = torch.cat( (allboxes, boxes.unsqueeze(0)), dim=0)
      allcls =  torch.cat( (allcls, cls.unsqueeze(0)   ), dim=0)
      
  return (allboxes.cpu().tolist(), allcls.tolist() )


if __name__=='__main__':

  fn = 'somepascalvoc/2007_001430.jpg'
  runstuff_objdet_singlefile(fn)
  
  #fnlist= ['./somepascalvoc/2007_001430.jpg','./somepascalvoc/2007_001594.jpg','./somepascalvoc/2007_001678.jpg']
  #runstuff_objdet_filelist(fnlist)
  
  
