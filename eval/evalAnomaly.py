# Copyright (c) OpenMMLab. All rights reserved.
import os
import torch
import random
from PIL import Image
import numpy as np
from torch.cuda import device_count
from tqdm import tqdm


from torchvision.transforms import Compose, CenterCrop, Normalize, Resize, ToTensor
import os.path as osp
from argparse import ArgumentParser
from ood_metrics import fpr_at_95_tpr, calc_metrics, plot_roc, plot_pr,plot_barcode
from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_recall_curve, average_precision_score
import torch.nn.functional as F
from transform import ToLabel

import sys
sys.path.append("..")
from models.erfnet import ERFNet
from models.enet import ENet
from models.bisenetv1 import BiSeNetV1

seed = 42

# general reproducibility
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

NUM_CHANNELS = 3
NUM_CLASSES = 20
# gpu training specific
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True






def get_softmax_t(y, temp=1.0):
    
    y /= temp        
    probs = np.array(F.softmax(torch.tensor(y), dim=0))

    return probs

def get_entropy(y):

    probs = get_softmax_t(y)
    entropy = - np.sum(probs * np.log(probs), axis=0) / np.log(probs.shape[0])
  
    return entropy

def mainEval(args):
    
   
    anomaly_score_list = []
    ood_gts_list = []

    if not os.path.exists('results.txt'):
        open('results.txt', 'w').close()
    file = open('results.txt', 'a')

    temp = float(args.temp)
    anomalyScore = args.anomalyScore
    modelpath = args.loadDir + args.loadModel
    weightspath = args.loadDir + args.loadWeights
    modelVersion = args.modelVersion
    void = args.void

    print ("Loading model: " + modelpath)
    print ("Loading weights: " + weightspath)

    model = None
    if void:
      if modelVersion == 'ERFNet':
        model = ERFNet(NUM_CLASSES)
      elif modelVersion == 'ENet':
        model = ENet(NUM_CLASSES)
      elif modelVersion == 'BiSeNetV1':
        model = BiSeNetV1(NUM_CLASSES)
      else:
        assert False, "wrong model version"
      
    else:
      model = ERFNet(NUM_CLASSES)

    device = 'cuda'
    if args.cpu:
      device = 'cpu'

   
    if (not args.cpu):
      model = torch.nn.DataParallel(model).cuda()
    else:
      model = model.cpu()

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
    
    print("Model and weights LOADED successfully")

    if args.resize:
      input_transform = Compose([
        Resize((512, 1024), Image.BILINEAR),
        ToTensor()
      ]) #changed according to eval_iou
      label_transform = Compose([
        Resize((512, 1024), Image.NEAREST),
        ToLabel()
      ])

    else:
      #NOTE: problem in build of BiSeNet some resolution cannot be downscaled to
      #ex. 780/16 = 48.75
      #    780/32 = 24.375
      #which means size problem
      #trick resize to nearest multiple of 32

      input_transform = Compose([
        ToTensor()
      ])
      label_transform = Compose([
        ToLabel()
      ])

      if modelVersion == 'BiSeNetV1':
        
        closest_size = (1024, 2048)

        if "RoadAnomaly21" in args.input[0]:
          closest_size = (736, 1280)

        if "RoadObsticle21" in args.input[0]:
          closest_size = (1088, 1920)
                       
        if "RoadAnomaly" in args.input[0]:
          closest_size = (736, 1280)

        input_transform = Compose([
          Resize(closest_size, Image.BILINEAR),
          ToTensor()
        ])
        label_transform = Compose([
          Resize(closest_size, Image.NEAREST),
          ToLabel()
        ])
    
    
    model.eval()    
    
    for path in tqdm(args.input):
        
        image = Image.open(path).convert('RGB')
        image = input_transform(image).unsqueeze(0).to(device)

        y = model(image)
        
        if modelVersion == 'BiSeNetV1':
          y=y[0]


        y = y.data.cpu().numpy()[0].astype("float32") #same as squeeze, remove batch

        if void:
            anomalyScore =  get_softmax_t(y)
            anomaly_result = anomalyScore[-1,:,:]
        else:
          if anomalyScore == 'MSP':
            probs = get_softmax_t(y[:-1], temp=temp)
            anomaly_result = 1 - np.max(probs, axis=0)
          elif anomalyScore == 'maxLogit':
            anomaly_result = - np.max(y[:-1], axis=0)
          elif anomalyScore == 'entropy':
            anomaly_result = get_entropy(y[:-1])
          

        pathGT = path.replace("images", "labels_masks")                
        if "RoadObsticle21" in pathGT:
          pathGT = pathGT.replace("webp", "png")
        if "fs_static" in pathGT:
          pathGT = pathGT.replace("jpg", "png")                
        if "RoadAnomaly" in pathGT:
          pathGT = pathGT.replace("jpg", "png")  

        mask = Image.open(pathGT).convert('L')
        mask = label_transform(mask).squeeze(0) #no unsqueeze (no batch needed for numpy)
        ood_gts = np.array(mask).astype('int32')

        #map to: 0=not anomaly, 1=anomaly
        
        if "RoadAnomaly" in pathGT:
          ood_gts = np.where((ood_gts==2), 1, ood_gts) # void value = 2 -> mapped to anomaly-not-obstacle
        

        if 1 not in np.unique(ood_gts):
            continue #anomaly not found in image
        else:
            ood_gts_list.append(ood_gts)
            anomaly_score_list.append(anomaly_result)
        
        del anomaly_result, ood_gts, mask
        torch.cuda.empty_cache()

        
    file.write( "\n")

    ood_gts = np.array(ood_gts_list)
    anomaly_scores = np.array(anomaly_score_list)
    
    ood_mask = (ood_gts == 1) #anomaly
    ind_mask = (ood_gts == 0) #not anomaly
    
    ood_out = anomaly_scores[ood_mask] #select only certain anomaly scores (output) (corresponding to ood_mask)
    ind_out = anomaly_scores[ind_mask] #select only certain anomaly scores (output) (corresponding to ind_mask)
   
    ood_label = np.ones(len(ood_out))
    ind_label = np.zeros(len(ind_out))
    
    val_out = np.concatenate((ind_out, ood_out))
    val_label = np.concatenate((ind_label, ood_label))
    
    prc_auc = average_precision_score(val_label, val_out)
    fpr = fpr_at_95_tpr(val_out, val_label)

    print(f'AUPRC score: {prc_auc*100.0}')
    print(f'FPR@TPR95: {fpr*100.0}')

    file.write(("args:" + str(args)))
    file.write(('    AUPRC score:' + str(prc_auc*100.0) + '   FPR@TPR95:' + str(fpr*100.0) ))
    file.close()

if __name__ == '__main__':
    
    parser = ArgumentParser()
    parser.add_argument(
        "--input",
        nargs="+",
        default="../datasets/Validation_Dataset/RoadAnomaly21/images/*.png",
        help="A directory where image are stored with n_image.ext",
    )  
    parser.add_argument('--loadDir',default="../trained_models/")
    parser.add_argument('--loadWeights', default="erfnet_pretrained.pth")
    parser.add_argument('--loadModel', default="../models/erfnet.py")
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--anomalyScore', default='maxLogit')
    parser.add_argument('--temp', default=1.0)
    parser.add_argument('--modelVersion', default='ERFNet')
    parser.add_argument('--void', action='store_true', help='Set this flag to activate')
    parser.add_argument('--resize', action='store_true')

    args = parser.parse_args()
    mainEval(args)