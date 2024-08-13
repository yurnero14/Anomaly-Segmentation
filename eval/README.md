# Functions for evaluating/visualizing the network's output

Currently there are 4 usable functions to evaluate stuff:
- eval_cityscapes_color
- eval_iou
- eval_forwardTime

## evalAnomaly.py

This code can be used to produce anomaly segmentation results on various anomaly metrics.

## eval_iou.py 

This code can be used to calculate the IoU (mean and per-class) in a subset of images with labels available, like Cityscapes val/train sets.

## eval_forwardTime.py
This function loads a model specified by '-m' and enters a loop to continuously estimate forward pass time (fwt) in the specified resolution. 

## Example
Usage example in "../Colab Notebook/StartScript_AML.ipynb"


