#Copyright (c) OpenMMLab. All rights reserved.
import os
import torch
import random
from PIL import Image
import numpy as np
from torch.cuda import device_count
from tqdm import tqdm
from torch import nn, optim


from torchvision.transforms import Compose, CenterCrop, Normalize, Resize
import os.path as osp
from argparse import ArgumentParser, Namespace
from ood_metrics import fpr_at_95_tpr, calc_metrics, plot_roc, plot_pr,plot_barcode
from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_recall_curve, average_precision_score
from torchvision.transforms import ToTensor
import torch.nn.functional as F

from transform import ToLabel
from evalAnomaly import mainEval
from torch.nn import BCELoss

import sys
sys.path.append("..")
from models.erfnet import ERFNet
from models.enet import ENet
from models.bisenetv1 import BiSeNetV1


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()

        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

        self.bce = BCELoss(reduction='none')

    def forward(self, inputs, targets):

        ce_loss = self.bce(inputs, targets)

        p_t = inputs * targets + (1 - inputs) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** self.gamma)
        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss
        if self.reduction == "none":
            return loss
        elif self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            raise ValueError(f"Invalid value for reduction: '{self.reduction}'")



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

datasets, formats, n_imgs =  ['RoadAnomaly21', 'RoadObsticle21', 'FS_LostFound_full', 'fs_static', 'RoadAnomaly'], ['png', 'webp', 'png', 'jpg', 'jpg'], [10, 30, 100, 30, 60]

n_pixels = [1280*720*10, 1920*1080*10/3, 2048*1024, 2048*1024*10/3, 1280*720*10/6] #loss weight
anomaly_non_anomaly = [0.143/0.822, 0.002/0.307, 0.002/0.811, 0.012/0.836, 0.098/0.902] #loss cost

#manual no standard dataloader
def generate_random_groups(population, group_size, num_groups):
    
    all_groups = []
    for _ in range(num_groups):
        
        random.shuffle(population)
        group = random.sample(population, group_size)
        all_groups.append(group)
        for i in group:
            population.remove(i)

    return all_groups

def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--input",
        default="../datasets/Validation_Dataset/",
        help="A directory where datasets are stored",
    )  
    parser.add_argument('--loadDir',default="../trained_models/")
    parser.add_argument('--loadWeights', default="erfnet_pretrained.pth")
    parser.add_argument('--loadModel', default="erfnet.py")
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--temp', default=1.0)
    parser.add_argument('--modelVersion', default='ERFNet')
    parser.add_argument("--saveT", default="")
    parser.add_argument("--num-epochs", default=5)
    parser.add_argument("--t0", default=1.1)
    args = parser.parse_args()
  

    
    temp = float(args.temp)
    modelpath = args.loadModel
    weightspath = args.loadDir + args.loadWeights
    modelVersion = args.modelVersion
    
    print ("Loading model: " + modelpath)
    print ("Loading weights: " + weightspath)


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

    
    input_transform = Compose([
      ToTensor()
    ])
    label_transform = Compose([
      ToLabel()
    ])
    
    model.eval() 
    
    initial_t = float(args.t0)

    #must be a learanble param
    temp = nn.Parameter(torch.tensor(initial_t).to(device),requires_grad=True)

    #find good lr
    #probably high since maxLogit is shown to perform better
    optimizer = optim.Adam([temp], lr=100.0)

    criterion = nn.BCELoss(reduction='mean')

    for epoch in range(int(args.num_epochs)):

        print(f"epoch: {epoch}")
        
        random_indexes = [ #max size without crash 46
          generate_random_groups([i for i in range(10)], 1, 10), #random_index_RA21
          generate_random_groups([i for i in range(30)], 3, 10), #random_index_RO21
          generate_random_groups([i for i in range(100)], 10, 10), #random_index_FSLF 
          generate_random_groups([i for i in range(30)], 3, 10), # random_index_FSstatic
          generate_random_groups([i for i in range(60)], 6, 10) # random_index_RA
        ]

        all_paths = []

        for i in range(10):

            pseudo_batch_fns = []#filenames of pseudobatches

            for j,(dataset,format) in enumerate(zip(datasets,formats)):
                for index in random_indexes[j][i]: #index of a dataset of a certain split
                    pseudo_batch_fns.append(args.input + f"{dataset}/images/{index}.{format}")

            random.shuffle(pseudo_batch_fns)

            all_paths.append(pseudo_batch_fns)

        print("\n--- REGRESSING BEST T ---")

        for pseudo_batch in tqdm(all_paths , desc="   batch"): #pick 115 images, we cannot do a classical batch but we need to stabilize (a lot!) training
            loss = 0.0

            for path in pseudo_batch:

                image = Image.open(path).convert('RGB')
                image = input_transform(image).unsqueeze(0).to(device)

                with torch.no_grad():
                    y = model(image)

                y = y[0]
                
                y = y / temp        
                probs = F.softmax(y, dim=0)
                anomaly_result = probs.max(dim=0)[0]
                
                pathGT = path.replace("images", "labels_masks")                
                if "RoadObsticle21" in pathGT:
                    pathGT = pathGT.replace("webp", "png")
                if "fs_static" in pathGT:
                    pathGT = pathGT.replace("jpg", "png")                
                if "RoadAnomaly" in pathGT:
                    pathGT = pathGT.replace("jpg", "png")  

                mask = Image.open(pathGT).convert('L')
                mask = label_transform(mask).squeeze(0) #no unsqueeze (no batch needed for numpy)
                ood_gts = mask.to(device)

                #map to: 0=not anomaly, 1=anomaly
                
                if "RoadAnomaly" in pathGT:
                    ood_gts = torch.where((ood_gts==2), 1, ood_gts) # void value = 2 -> mapped to anomaly-not-obstacle
                
                
                ood_gts_flat = ood_gts.flatten()
                anomaly_result_flat = anomaly_result.flatten()

                an_filtered = anomaly_result_flat[ood_gts_flat < 2]
                ood_gt_filtered = ood_gts_flat[ood_gts_flat < 2]

                # Assuming you have true labels (y_true) and predicted labels (y_pred)
                fpr, tpr, thresholds = roc_curve(ood_gt_filtered.cpu().detach().numpy().astype(np.int32), an_filtered.cpu().detach().numpy())

                # Find the index where TPR is closest to 95%
                tpr_idx = np.argmin(np.abs(tpr - 0.95))

                an_ood_gt_sorted, _ = torch.sort(torch.stack([an_filtered, ood_gt_filtered],dim=1), dim=0)
                
                an_filtered95 = an_ood_gt_sorted[tpr_idx:,0]
                ood_gt_filtered95 = an_ood_gt_sorted[tpr_idx:,1]

                FP_samples = an_filtered95[ood_gt_filtered95==0]

                 
                for i,ds_name in enumerate(datasets):
                    if ds_name in path:
                        #optimize overall performance (AUPRC), ensures not classify everithing as 0
                        fit_loss = FocalLoss(alpha=anomaly_non_anomaly[i], reduction="mean")
                        #optimize FPR@TPR95
                        FP_loss=  FP_samples.sum()/10000 #sum of max(Pclass)

                        loss += (FP_loss)*((2048*1024)/n_pixels[i])
                        break

                
                
                del mask,ood_gts,ood_gts_flat,anomaly_result,anomaly_result_flat,y
                
                torch.cuda.empty_cache()
            
            loss.backward()
            optimizer.step()  # Update temp
            
            optimizer.zero_grad()

            print(f"loss: {loss}, temp:{temp.item()}")     
          
            del loss
            torch.cuda.empty_cache()

        
      
    
    
        print("--- VALIDATING EPOCH TEMP ---")
      
        for dataset,format,n_img in zip(datasets, formats, n_imgs):

            print(f"dataset:{dataset}, MSP temperature:{temp.item()}, device:{device}, resize to 512x1024:{False}") 
               
            mainEval(Namespace(
              input=[f"{args.input}/{dataset}/images/{i}.{format}" for i in range(n_img)],
              temp=f"{temp.item()}",
              anomalyScore="MSP",
              loadDir="../trained_models/",
              loadWeights="erfnet_pretrained.pth",
              loadModel="erfnet.py",
              modelVersion="ERFNet",
              cpu=device=='cpu',
              void=False,
              resize=False
            ))


    

if __name__ == '__main__':
    main()