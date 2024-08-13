# Main code for training ERFNet model in Cityscapes dataset
# Sept 2017
# Eduardo Romera
#######################

import os
import random
import time
import numpy as np
import torch
import math


from PIL import Image, ImageOps
from argparse import ArgumentParser

from torch.optim import SGD, Adam, lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, CenterCrop, Normalize, Resize, Pad
from torchvision.transforms import ToTensor, ToPILImage

from dataset import cityscapes
from transform import Relabel, ToLabel, Colorize, TransformationTrain

from loss import OhemCELoss
from IsoMaxPlus.isomaxplus import *
from bisenet_optim import set_optimizer
from bisenet_scheduler import WarmupPolyLrScheduler
from dist_sampler import RepeatedDistSampler

import importlib
from iouEval import iouEval, getColorEntry

import torch.nn as nn
from shutil import copyfile

NUM_CHANNELS = 3
NUM_CLASSES = 20 #pascal=22, cityscapes=20

color_transform = Colorize(NUM_CLASSES)
image_transform = ToPILImage()

import sys
from models.bisenetv1 import BiSeNetV1
from models.enet import ENet
from models.erfnetIsoMax import ERFNet
from models.erfnet_imagenet import ERFNet as ERFNet_imagenet


from torch.utils.tensorboard import SummaryWriter

#Augmentations - different function implemented to perform random augments on both image and target
class MyCoTransform(object):
    def __init__(self, enc, augment=True, height=512, add_transform=None):
        self.enc=enc
        self.augment = augment
        self.height = height
        self.add_transform = add_transform
        pass

    def __call__(self, input, target):
        # do something to both images
        input =  Resize(self.height, Image.BILINEAR)(input)
        target = Resize(self.height, Image.NEAREST)(target)
            
        
        if self.add_transform is None:
            
            if(self.augment):
                # Random hflip
                hflip = random.random()
                if (hflip < 0.5):
                    input = input.transpose(Image.FLIP_LEFT_RIGHT)
                    target = target.transpose(Image.FLIP_LEFT_RIGHT)
                
                #Random translation 0-2 pixels (fill rest with padding #what purpose?
                transX = random.randint(-2, 2) 
                transY = random.randint(-2, 2)

                input = ImageOps.expand(input, border=(transX,transY,0,0), fill=0)
                target = ImageOps.expand(target, border=(transX,transY,0,0), fill=255) #pad label filling with 255
                input = input.crop((0, 0, input.size[0]-transX, input.size[1]-transY))
                target = target.crop((0, 0, target.size[0]-transX, target.size[1]-transY))   

            if (self.enc):
                target = Resize(int(self.height/8), Image.NEAREST)(target)
            
        else:
            # lot of augmentation
            im_lb = self.add_transform({'im':np.array(input), 'lb':np.array(target)})
            
            input = im_lb['im']
            target = im_lb['lb']

        input = ToTensor()(input)
        target = ToLabel()(target)
        target = Relabel(255, 19)(target)
        
        return input, target


class CrossEntropyLoss2d(torch.nn.Module):

    def __init__(self, weight=None):
        super().__init__()

        self.loss = torch.nn.NLLLoss(weight)

    def forward(self, outputs, targets):
        return self.loss(torch.nn.functional.log_softmax(outputs, dim=1), targets)


def train(args, model, enc=False):
    best_acc = 0

    #TODO: calculate weights by processing dataset histogram (now its being set by hand from the torch values)
    #create a loder to run all images and calculate histogram of labels, then create weight array using class balancing

    weight = torch.ones(NUM_CLASSES)
    if (enc):
        weight[0] = 2.3653597831726	
        weight[1] = 4.4237880706787	
        weight[2] = 2.9691488742828	
        weight[3] = 5.3442072868347	
        weight[4] = 5.2983593940735	
        weight[5] = 5.2275490760803	
        weight[6] = 5.4394111633301	
        weight[7] = 5.3659925460815	
        weight[8] = 3.4170460700989	
        weight[9] = 5.2414722442627	
        weight[10] = 4.7376127243042	
        weight[11] = 5.2286224365234	
        weight[12] = 5.455126285553	
        weight[13] = 4.3019247055054	
        weight[14] = 5.4264230728149	
        weight[15] = 5.4331531524658	
        weight[16] = 5.433765411377	
        weight[17] = 5.4631009101868	
        weight[18] = 5.3947434425354
    else:
        weight[0] = 2.8149201869965	
        weight[1] = 6.9850029945374	
        weight[2] = 3.7890393733978	
        weight[3] = 9.9428062438965	
        weight[4] = 9.7702074050903	
        weight[5] = 9.5110931396484	
        weight[6] = 10.311357498169	
        weight[7] = 10.026463508606	
        weight[8] = 4.6323022842407	
        weight[9] = 9.5608062744141	
        weight[10] = 7.8698215484619	
        weight[11] = 9.5168733596802	
        weight[12] = 10.373730659485	
        weight[13] = 6.6616044044495	
        weight[14] = 10.260489463806	
        weight[15] = 10.287888526917	
        weight[16] = 10.289801597595	
        weight[17] = 10.405355453491	
        weight[18] = 10.138095855713	

    weight[19] = 1.0 #0.0
      

    assert os.path.exists(args.datadir), "Error: datadir (dataset directory) could not be loaded"

    #choose input transform

    if args.model == "bisenetv1":
      add_transform = TransformationTrain(scales=[1.0, 1.25, 1.5, 1.75, 2.0], cropsize=[args.height, args.height])
    else:
      add_transform = None
    
    co_transform = MyCoTransform(enc, augment=True, height=args.height, add_transform=add_transform)
    co_transform_val = MyCoTransform(enc, augment=False, height=args.height)
    
    dataset_train = cityscapes(args.datadir, co_transform, 'train')
    dataset_val = cityscapes(args.datadir, co_transform_val, 'val')

    if args.model == "bisenetv1":
      sampler = RepeatedDistSampler(dataset_train, 1000*args.num_epochs*args.batch_size, shuffle=True)
      loader = DataLoader(dataset_train, num_workers=args.num_workers, batch_size=args.batch_size, sampler=sampler)
      iter_loader=iter(loader)
      loader_val = DataLoader(dataset_val, num_workers=args.num_workers, batch_size=1, shuffle=False)
    else:
      loader = DataLoader(dataset_train, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=True)
      loader_val = DataLoader(dataset_val, num_workers=args.num_workers, batch_size=1, shuffle=False)
    

    if args.cuda:
        weight = weight.cuda()
    
    savedir = f'./save/{args.savedir}'

    if (enc):
        automated_log_path = savedir + "/automated_log_encoder.txt"
        modeltxtpath = savedir + "/model_encoder.txt"
    else:
        automated_log_path = savedir + "/automated_log.txt"
        modeltxtpath = savedir + "/model.txt"    

    if (not os.path.exists(automated_log_path)):    #dont add first line if it exists 
        with open(automated_log_path, "a") as myfile:
            myfile.write("Epoch\t\tTrain-loss\t\tTest-loss\t\tTrain-IoU\t\tTest-IoU\t\tlearningRate")

    with open(modeltxtpath, "w") as myfile:
        myfile.write(str(model))

    #choose loss, optimizer, scheduler

    if args.model == "erfnetIsoMax":
      criterion =IsoMaxPlusLossSecondPart(entropic_scale=10.0)
      optimizer = Adam(model.parameters(), 5e-4, (0.9, 0.999),  eps=1e-08, weight_decay=1e-4)
      lambda1 = lambda epoch: pow((1-((epoch-1)/args.num_epochs)),0.9)  
      scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1) 

    elif args.model == "enet":
      criterion = CrossEntropyLoss2d(weight)
      optimizer = Adam(model.parameters(), 5e-4, (0.9, 0.999),  eps=1e-08, weight_decay=1e-4)
      lambda1 = lambda epoch: pow((1-((epoch-1)/args.num_epochs)),0.9)  ## scheduler 2
      scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)                             ## scheduler 2

    elif args.model == "bisenetv1":
      criteria_p = OhemCELoss(thresh=0.7)
      criteria_16 = OhemCELoss(thresh=0.7)
      criteria_32 = OhemCELoss(thresh=0.7)
      optimizer = set_optimizer(model)
      #NOTE: use a iter scheduler
      scheduler = WarmupPolyLrScheduler(optimizer, power=2,
        max_iter=80000, warmup_iter=1000,
        warmup_ratio=1, warmup='exp', last_epoch=-1,)
      #tot_iter = args.num_epochs*1000
      #lambda1 = lambda step: pow((1-(step/tot_iter)),0.9)  
      #scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1) 
      
    else:
      assert False, "model not defined"

    start_epoch = 1

    if args.resume: 
        if enc:
            filenameCheckpoint = savedir + '/checkpoint_enc.pth.tar'
        else:
            filenameCheckpoint = savedir + '/checkpoint.pth.tar'

        assert os.path.exists(filenameCheckpoint), "Error: resume option was used but checkpoint was not found in folder"
        checkpoint = torch.load(filenameCheckpoint)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        best_acc = checkpoint['best_acc']
        print("=> Loaded checkpoint at epoch {})".format(checkpoint['epoch']))

    
    writer = SummaryWriter(f"./runs/{args.model}")

    step_bisenet=0

    for epoch in range(start_epoch, args.num_epochs+1):
        print("----- TRAINING - EPOCH", epoch, "-----")

        epoch_loss = []
        time_train = []
     
        doIouTrain = args.iouTrain   
        doIouVal =  args.iouVal      

        if (doIouTrain):
            iouEvalTrain = iouEval(NUM_CLASSES)


        model.train() 
        if args.model != "bisenetv1":
          
          usedLr = 0
          for param_group in optimizer.param_groups:
              print("LEARNING RATE: ", param_group['lr'])
              usedLr = float(param_group['lr'])
          
          for step, (images, labels) in enumerate(loader):

              start_time = time.time()
              
              if args.cuda:
                  images = images.cuda()
                  labels = labels.cuda()

              inputs = Variable(images)
              targets = Variable(labels)

              #forward pass
              if args.model == "erfnetIsoMax":
                outputs = model(inputs, only_encode=enc)
              elif args.model == "enet":
                outputs = model(inputs)
              else:
                assert False, "model not found"  
            
              optimizer.zero_grad()
              #loss calculation
              loss = criterion(outputs, targets[:, 0])#erfnet and enet doesn't have intermediate losses
              
              loss.backward()
              optimizer.step()
                
              epoch_loss.append(loss.data.item()) #modified: loss.data[0] -> loss.data.item()
              time_train.append(time.time() - start_time)

              if (doIouTrain):
                  iouEvalTrain.addBatch(outputs.max(1)[1].unsqueeze(1).data, targets.data)
                  
              if args.visualize and args.steps_plot > 0 and step % args.steps_plot == 0:
                  start_time_plot = time.time()
                  image = inputs[0].cpu().data
                  
              if args.steps_loss > 0 and step % args.steps_loss == 0:
                  average = sum(epoch_loss) / len(epoch_loss)
                  print(f'loss: {average:0.4} (epoch: {epoch}, step: {step})', 
                          "// Avg time/img: %.4f s" % (sum(time_train) / len(time_train) / args.batch_size))

          scheduler.step(epoch) 
    
        else: 
          usedLr=0
          lr = scheduler.get_lr()
          usedLr = sum(lr) / len(lr)
          print("LEARNING RATE: ", usedLr)

          for step in range(1000):
             
              images, labels = next(iter_loader)

              start_time = time.time()
              
              if args.cuda:
                  images = images.cuda()
                  labels = labels.cuda()

              inputs = Variable(images)
              targets = Variable(labels)

              #forward pass
              if args.model == "bisenetv1":
                out, *h16_32 = model(inputs)
              else:
                assert False, "model not found"  
            
              optimizer.zero_grad()
              #loss calculation
              loss = criteria_p(out, targets[:, 0]) + criteria_16(h16_32[0], targets[:, 0]) + criteria_32(h16_32[1], targets[:, 0])
              
              loss.backward()
              optimizer.step() #lr managed only by scheduler for bisenet
              scheduler.step() 
              #scheduler.step(epoch*1000+step)

              epoch_loss.append(loss.data.item()) #modified: loss.data[0] -> loss.data.item()
              time_train.append(time.time() - start_time)

              if (doIouTrain):
                  iouEvalTrain.addBatch(outputs.max(1)[1].unsqueeze(1).data, targets.data)
                  
              if args.visualize and args.steps_plot > 0 and step % args.steps_plot == 0:
                  start_time_plot = time.time()
                  image = inputs[0].cpu().data
                  
              if args.steps_loss > 0 and step % args.steps_loss == 0:
                  average = sum(epoch_loss) / len(epoch_loss)
                  print(f'loss: {average:0.4} (epoch: {epoch}, step: {step})', 
                          "// Avg time/img: %.4f s" % (sum(time_train) / len(time_train) / args.batch_size))




        
        average_epoch_loss_train = sum(epoch_loss) / len(epoch_loss)
        
        iouTrain = 0
        if (doIouTrain):
            iouTrain, iou_classes = iouEvalTrain.getIoU()
            iouStr = getColorEntry(iouTrain)+'{:0.2f}'.format(iouTrain*100) + '\033[0m'
            print ("EPOCH IoU on TRAIN set: ", iouStr, "%")  

        #Validate on 500 val images after each epoch of training
        print("----- VALIDATING - EPOCH", epoch, "-----")
        model.eval()
        epoch_loss_val = []
        time_val = []

        if (doIouVal):
            iouEvalVal = iouEval(NUM_CLASSES)

        for step, (images, labels) in enumerate(loader_val):
            start_time = time.time()
            if args.cuda:
                images = images.cuda()
                labels = labels.cuda()

            inputs = Variable(images, volatile=True)    #volatile flag makes it free backward or outputs for eval
            targets = Variable(labels, volatile=True)
            
            with torch.no_grad():
              
              #forward pass
              if args.model == "erfnetIsoMax":
                outputs = model(inputs, only_encode=enc)
              elif args.model == "enet":
                outputs = model(inputs)
              elif args.model == "bisenetv1":
                outputs = model(inputs)[0] #no need for discard if you are validating

              #calculate losses (only to plot)
              if args.model == "bisenetv1":
                loss = criteria_p(outputs, targets[:, 0])
              else:
                loss = criterion(outputs, targets[:, 0])

            epoch_loss_val.append(loss.data.item()) #modified: loss.data[0] -> loss.data.item()
            time_val.append(time.time() - start_time)


            #Add batch to calculate TP, FP and FN for iou estimation
            if (doIouVal):
                iouEvalVal.addBatch(outputs.max(1)[1].unsqueeze(1).data, targets.data)
                
            if args.visualize and args.steps_plot > 0 and step % args.steps_plot == 0:
                start_time_plot = time.time()
                image = inputs[0].cpu().data
  
            if args.steps_loss > 0 and step % args.steps_loss == 0:
                average = sum(epoch_loss_val) / len(epoch_loss_val)
                print(f'VAL loss: {average:0.4} (epoch: {epoch}, step: {step})', 
                        "// Avg time/img: %.4f s" % (sum(time_val) / len(time_val) / args.batch_size))

        
        average_epoch_loss_val = sum(epoch_loss_val) / len(epoch_loss_val)

        iouVal = 0
        if (doIouVal):
            iouVal, iou_classes = iouEvalVal.getIoU()
            iouStr = getColorEntry(iouVal)+'{:0.2f}'.format(iouVal*100) + '\033[0m'
            print ("EPOCH IoU on VAL set: ", iouStr, "%") 
           

        # remember best valIoU and save checkpoint
        if iouVal == 0:
            current_acc = -average_epoch_loss_val
        else:
            current_acc = iouVal 
        is_best = current_acc > best_acc
        best_acc = max(current_acc, best_acc)
        if enc:
            filenameCheckpoint = savedir + '/checkpoint_enc.pth.tar'
            filenameBest = savedir + '/model_best_enc.pth.tar'    
        else:
            filenameCheckpoint = savedir + '/checkpoint.pth.tar'
            filenameBest = savedir + '/model_best.pth.tar'
        
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': str(model),
            'state_dict': model.state_dict(),
            'best_acc': best_acc,
            'optimizer' : optimizer.state_dict() if args.model != "bisenetv1" else None,
        }, is_best, filenameCheckpoint, filenameBest)

        #log loss and mIoU
        writer.add_scalars("Loss", {
                            'train':average_epoch_loss_train,
                            'val':average_epoch_loss_val
                          }, epoch)
        writer.add_scalar("mIoU", iouVal, epoch)

        #SAVE MODEL AFTER EPOCH
        if (enc):
            filename = f'{savedir}/model_encoder-{epoch:03}.pth'
            filenamebest = f'{savedir}/model_encoder_best.pth'
        else:
            filename = f'{savedir}/model-{epoch:03}.pth'
            filenamebest = f'{savedir}/model_best.pth'
        if args.epochs_save > 0 and step > 0 and step % args.epochs_save == 0:
            torch.save(model.state_dict(), filename)
            print(f'save: {filename} (epoch: {epoch})')
        if (is_best):
            torch.save(model.state_dict(), filenamebest)
            print(f'save: {filenamebest} (epoch: {epoch})')
            if (not enc):
                with open(savedir + "/best.txt", "w") as myfile:
                    myfile.write("Best epoch is %d, with Val-IoU= %.4f" % (epoch, iouVal))   
            else:
                with open(savedir + "/best_encoder.txt", "w") as myfile:
                    myfile.write("Best epoch is %d, with Val-IoU= %.4f" % (epoch, iouVal))           

        #SAVE TO FILE A ROW WITH THE EPOCH RESULT (train loss, val loss, train IoU, val IoU)
        #Epoch		Train-loss		Test-loss	Train-IoU	Test-IoU		learningRate
        with open(automated_log_path, "a") as myfile:
            myfile.write("\n%d\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.8f" % (epoch, average_epoch_loss_train, average_epoch_loss_val, iouTrain, iouVal, usedLr ))
    
    writer.close()

    return(model)   #return model (convenience for encoder-decoder training)

def save_checkpoint(state, is_best, filenameCheckpoint, filenameBest):
    torch.save(state, filenameCheckpoint)
    if is_best:
        print ("Saving model as best")
        torch.save(state, filenameBest)


def main(args):
    savedir = f'./save/{args.savedir}'

    if not os.path.exists(savedir):
        os.makedirs(savedir)

    with open(savedir + '/opts.txt', "w") as myfile:
        myfile.write(str(args))

    #Load Model
    assert os.path.exists("./models/"+ args.model + ".py"), "Error: model definition not found"
    model_file = importlib.import_module("models."+args.model)
    if args.model == "erfnetIsoMax":
      model = model_file.ERFNet(NUM_CLASSES)
    elif args.model == "enet":
      model = model_file.ENet(NUM_CLASSES)
    elif args.model == "bisenetv1": 
      model = model_file.BiSeNetV1(NUM_CLASSES)
    else:
      assert False, "model file not exist"
    
    copyfile("./models/" + args.model + ".py", savedir + '/' + args.model + ".py")
    
    if args.cuda:
        model = torch.nn.DataParallel(model).cuda()
    
    if args.state:
        #if args.state is provided then load this state for training
        #Note: this only loads initialized weights. If you want to resume a training use "--resume" option!!
        """
        try:
            model.load_state_dict(torch.load(args.state))
        except AssertionError:
            model.load_state_dict(torch.load(args.state,
                map_location=lambda storage, loc: storage))
        #When model is saved as DataParallel it adds a model. to each key. To remove:
        #state_dict = {k.partition('model.')[2]: v for k,v in state_dict}
        #https://discuss.pytorch.org/t/prefix-parameter-names-in-saved-model-if-trained-by-multi-gpu/494
        """
        def load_my_state_dict(model, state_dict):  #custom function to load model when not all dict keys are there
            own_state = model.state_dict()
            for name, param in state_dict.items():
                if name not in own_state:
                     continue
                own_state[name].copy_(param)
            return model

        #print(torch.load(args.state))
        model = load_my_state_dict(model, torch.load(args.state))


    if args.model == "erfnetIsoMax":
      if (not args.decoder):
          print("========== ENCODER TRAINING ===========")
          model = train(args, model, True) #Train encoder
      #CAREFUL: for some reason, after training encoder alone, the decoder gets weights=0. 
      #We must reinit decoder weights or reload network passing only encoder in order to train decoder
      print("========== DECODER TRAINING ===========")
      if (not args.state):
          if args.pretrainedEncoder:
              print("Loading encoder pretrained in imagenet")
              
              pretrainedEnc = torch.nn.DataParallel(ERFNet_imagenet(1000))
              pretrainedEnc.load_state_dict(torch.load(args.pretrainedEncoder)['state_dict'])
              pretrainedEnc = next(pretrainedEnc.children()).features.encoder
              if (not args.cuda):
                  pretrainedEnc = pretrainedEnc.cpu()     #because loaded encoder is probably saved in cuda
          else:
              pretrainedEnc = next(model.children()).encoder
          model = model_file.ERFNet(NUM_CLASSES, encoder=pretrainedEnc)  #Add decoder to encoder
          if args.cuda:
              model = torch.nn.DataParallel(model).cuda()
          #When loading encoder reinitialize weights for decoder because they are set to 0 when training dec
      model = train(args, model, False)   #Train decoder
    elif args.model == "enet":
      print("========== ENET TRAINING ===========")
      model = train(args, model, False) #no decoder to train separately
    elif args.model == "bisenetv1":
      print("========== BISENET TRAIN ===========")
      model = train(args, model, False) #no decoder to train separately
    print("========== TRAINING FINISHED ===========")

if __name__ == '__main__':
    parser = ArgumentParser()
    
    parser.add_argument('--cuda', action='store_true', default=True)  #NOTE: cpu-only has not been tested so you might have to change code if you deactivate this flag
    parser.add_argument('--model', default="erfnetIsoMax")
    parser.add_argument('--state')

    parser.add_argument('--datadir', default="")
    parser.add_argument('--height', type=int, default=512)
    parser.add_argument('--num-epochs', type=int, default=150)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--steps-loss', type=int, default=50)
    parser.add_argument('--steps-plot', type=int, default=50)
    parser.add_argument('--epochs-save', type=int, default=0)    #You can use this value to save model every X epochs
    parser.add_argument('--savedir', required=True)
    parser.add_argument('--decoder', action='store_true')
    parser.add_argument('--pretrainedEncoder') #, default="./trained_models/erfnet_encoder_pretrained.pth.tar")
    parser.add_argument('--visualize', action='store_true')

    parser.add_argument('--iouTrain', action='store_true', default=False) #recommended: False (takes more time to train otherwise)
    parser.add_argument('--iouVal', action='store_true', default=True)  
    parser.add_argument('--resume', action='store_true')    #Use this flag to load last checkpoint for training  


    main(parser.parse_args())
