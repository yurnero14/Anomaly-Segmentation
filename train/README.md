# Training ERFNet, ENet, BiSeNet in Pytorch 

PyTorch code for training ERFNet model on Cityscapes. The code was based initially on the code from [bodokaiser/piwise](https://github.com/bodokaiser/piwise), adapted with several custom added modifications and tweaks. Some of them are:
- Load cityscapes dataset
- ERFNet, Enet, BiSeNet model definition
- Calculate IoU on each epoch during training
- Save snapshots and best model during training
- Save additional output files useful for checking results (see below "Output files...")
- Resume training from checkpoint (use "--resume" flag in the command)

## Example

Usage example in "../Colab Notebooks/StartScript_AML.ipynb"
