# Imports here
import argparse
import torch
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, models, transforms
import torchvision
import numpy as np
import pandas as pd
from itertools import chain
from torch.autograd import Variable
from PIL import Image
import os, random
import matplotlib.pyplot as plt
import json

## accept input args
def get_input_args():
    '''
        Process input args from the user.
    '''
    parser = argparse.ArgumentParser(description='Create a trained neural network.')

    parser.add_argument('--input', type=str, default="", help='location of input image, if none is provide a random one will be selected.')
    parser.add_argument('--checkpoint', type=str, default="./checkpoint.pth", help='Location of where checkpoint is.')
    parser.add_argument('--top_k', type=int, default=5, help='Number of results returned.')
    parser.add_argument('--gpu', type=bool, default=False, help='gpu enabled.')
    parser.add_argument('--category_names', type=str, default="./cat_to_name.json", help='List of category names.')
    
    # extrac incase we need it
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-v", "--verbose", action="store_true")
    group.add_argument("-q", "--quiet", action="store_true")
    args = parser.parse_args()

    # Replace None with parser.parse_args() parsed argument collection that 
    # you created with this function 
    return args

# have a load in function for the checkpoint

def get_random_test_image_path():
    '''
        This function will take a random image from the test set.
    '''
    # Show picture
    root_test_dir = './flowers/test'
    test_folder = random.choice(os.listdir(root_test_dir+'/'))
    img = random.choice(os.listdir(root_test_dir+'/'+test_folder+'/'))
    img_path = root_test_dir+'/'+test_folder+'/'+ img
    print(img_path)
    return img_path

def load_checkpoint(filename):
    '''
        Reload the checkpoint.
    '''
    cuda = torch.cuda.is_available()

    if cuda:
        checkpoint = torch.load(filename)
    else:
        checkpoint = torch.load(filename, map_location='cpu')    
        
    learning_rate = checkpoint['learning_rate']
    model = getattr(models, checkpoint['arch'])(pretrained=True)
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.epochs = checkpoint['epochs']
    model.class_to_idx = checkpoint['class_to_idx']
        
    return model

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    data_size_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    img_pil = Image.open(image)

    img_tensor = data_size_transforms(img_pil)

    return img_tensor

def predict(image_path, model, gpu, topk=5):
    ''' 
        Predict the class (or classes) of an image using a trained deep learning model.
    '''
    cuda = torch.cuda.is_available()
    if cuda and gpu:
        # Move model parameters to the GPU
        model.cuda()
    else:
        model.cpu()
    
    # turn off dropout
    model.eval()

    # The image
    img_torch = process_image(image_path)
    img_torch = img_torch.unsqueeze_(0)
    img_torch = img_torch.float()
    
    with torch.no_grad():
        if cuda and gpu:
            output = model.forward(img_torch.cuda())
        else:
            output = model.forward(img_torch)
    
    probs, indices = torch.topk(F.softmax(output, dim=1), topk, sorted=True)

    idx_to_class = { v:k for k, v in model.class_to_idx.items()}
    return ([prob.item() for prob in probs[0].data], 
            [idx_to_class[ix.item()] for ix in indices[0].data])

def load_categories(filename):
    '''
        Load the passed in JSON filename.
    '''
    with open(filename) as f:
        category_names = json.load(f)
    return category_names

def main():
    args = get_input_args()
    
    model = load_checkpoint(args.checkpoint) #  
    
    cat_to_name = load_categories(args.category_names)   
    
    if args.input == "":
        image_location = get_random_test_image_path()
    
    probs, classes = predict(image_location, model, args.gpu, topk=args.top_k)
    
    print(probs, [cat_to_name[name] for name in classes])

# Call to main function to run the program
if __name__ == "__main__":
    main()
    