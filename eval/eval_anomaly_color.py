# Code to produce colored segmentation output in Pytorch for all cityscapes subsets  
# Sept 2017
# Eduardo Romera
#######################

import numpy as np
import torch
import os
import importlib

from PIL import Image
from argparse import ArgumentParser

from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, CenterCrop, Normalize, Resize
from torchvision.transforms import ToTensor, ToPILImage


from erfnet import ERFNet
from transform import Relabel, ToLabel, Colorize


#print example from datasets to check if are correct (do color remapping)
import matplotlib.pyplot as plt


NUM_CHANNELS = 3
NUM_CLASSES = 20


def load_image(file):
    return Image.open(file)



cityscapes_trainIds2labelIds = Compose([
    Relabel(19, 255),  
    Relabel(18, 33),
    Relabel(17, 32),
    Relabel(16, 31),
    Relabel(15, 28),
    Relabel(14, 27),
    Relabel(13, 26),
    Relabel(12, 25),
    Relabel(11, 24),
    Relabel(10, 23),
    Relabel(9, 22),
    Relabel(8, 21),
    Relabel(7, 20),
    Relabel(6, 19),
    Relabel(5, 17),
    Relabel(4, 13),
    Relabel(3, 12),
    Relabel(2, 11),
    Relabel(1, 8),
    Relabel(0, 7),
    Relabel(255, 0),
    ToPILImage(),
])



def main(args):

    device = 'cuda'
    if args.cpu:
        device = 'cpu'

    if args.no_resize:
        input_transform = Compose([
            ToTensor(),
            #Normalize([.485, .456, .406], [.229, .224, .225]),
        ])
        

    else:
        input_transform = Compose([
            Resize((512,1024),Image.BILINEAR),
            ToTensor(),
            #Normalize([.485, .456, .406], [.229, .224, .225]),
        ])
        

    modelpath = args.loadModel
    weightspath = args.loadDir + args.loadWeights

   
    print ("Loading model: " + modelpath)
    print ("Loading weights: " + weightspath)

    #Import ERFNet model from the folder
    #Net = importlib.import_module(modelpath.replace("/", "."), "ERFNet")
    model = ERFNet(NUM_CLASSES)
  
    if (not args.cpu):
        model = torch.nn.DataParallel(model).cuda()
    else:
        model = model.cpu()

    #model.load_state_dict(torch.load(args.state))
    #model.load_state_dict(torch.load(weightspath)) #not working if missing key

    def load_my_state_dict(model, state_dict): 
        own_state = model.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                if name.startswith("module."):
                    own_state[name.split("module.")[-1]].copy_(param)
                else:
                    print(name, " not loaded")
                    continue
            else:
                own_state[name].copy_(param)
        return model

    model = load_my_state_dict(model, torch.load(weightspath, map_location=lambda storage, loc: storage))
    
    print ("Model and weights LOADED successfully")

    model.eval()

    if(not os.path.exists(args.datadir)):
        print ("Error: datadir could not be loaded")

    fig, ax = plt.subplots(5, 2, figsize=(15, 5))
    plt.subplots_adjust(top=2.5,bottom=0.0)  # Adjust this value as needed, is in inches
         
    for i,(dataset,format) in enumerate(zip(['RoadAnomaly21', 'RoadObsticle21', 'FS_LostFound_full', 'fs_static', 'RoadAnomaly'], ['png', 'webp', 'png', 'jpg', 'jpg'])):
        
        image_path = os.path.join(args.datadir, f'{dataset}/images/0.{format}')
        image=input_transform(load_image(image_path)).unsqueeze(0)

        if (not args.cpu):
          image = image.cuda()
          
        with torch.no_grad():
          outputs = model(image)

        label = outputs[0].max(0)[1].byte().cpu().data
            
        label_color = Colorize()(label.unsqueeze(0))
        maxLogit = -outputs[0][:-1].max(0)[0]
        #label_cityscapes = cityscapes_trainIds2labelIds(label.unsqueeze(0))
        
           
            
        ax[i][0].imshow(image[0,...].permute(1,2,0))
        ax[i][0].set_title(f"image {dataset}")
        ax[i][1].imshow(maxLogit,cmap='coolwarm')
        ax[i][1].set_title(f"maxLogit {dataset}")
            

    plt.show()

    

if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--state')

    parser.add_argument('--loadDir',default="../trained_models/")
    parser.add_argument('--loadWeights', default="erfnet_pretrained.pth")
    parser.add_argument('--loadModel', default="erfnet.py")
    parser.add_argument('--subset', default="val")  #can be val, test, train, demoSequence

    parser.add_argument('--datadir')
    parser.add_argument('--num-workers', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--no-resize', action='store_true')

    main(parser.parse_args())