# standard imports
import argparse
from time import time
from PIL import Image
import os, random
import matplotlib.pyplot as plt
# from itertools import chain
from collections import OrderedDict

# pytorch imports
import torch
from torch import optim
from torch import nn
from torchvision import datasets, transforms
import torchvision.models as models
from torch.autograd import Variable

def get_input_args():
    '''
        Process input args from the user.
    '''
    parser = argparse.ArgumentParser(description='Create a trained neural network.')

    parser.add_argument('--data_directory', type=str, default="./flowers", help='Directory of where the images folders are located (train,test and valid).')
    parser.add_argument('--arch', type=str, default="densenet161", help='architecture type, [vgg19, densenet161]')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate is the rate at which the weights update.')
    parser.add_argument('--hidden_units', type=int, default=(512 * 7 * 7), help='hidden_units.')
    parser.add_argument('--epochs', type=int, default=7, help='number of repeated training sets.')
    parser.add_argument('--save_dir', type=str, default="./", help='Directory where you would like the checkpoint to be saved.')
    parser.add_argument('--checkpoint', type=bool, default=False,help='Save trained model checkpoint to file')
    parser.add_argument('--gpu', type=bool, default=True, help='gpu enabled.')
    
    # extrac incase we need it
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-v", "--verbose", action="store_true")
    group.add_argument("-q", "--quiet", action="store_true")
    args = parser.parse_args()

    # Replace None with parser.parse_args() parsed argument collection that 
    # you created with this function 
    return args

def get_dirs(args) -> dict:
    '''
        Loads the training data directories
    '''
    dirs = { 'train': args.data_directory+'/train',
             'test': args.data_directory+'/test',
             'valid': args.data_directory+'/valid'}
    
    for key in dirs:
        if not os.path.isdir(dirs[key]): 
            raise Exception('Not a Directory! '+dirs[key])
            
    return dirs
        
def get_dataloaders(dirs: dict):
    '''
        Transforms the data for each of the images to be processed, and creates a dict object with the train, test, and validation datasets.
    '''
    size = 224
    data_size_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    ])

    data_transforms = transforms.Compose([
        transforms.RandomRotation(45),
        transforms.RandomResizedCrop(size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    image_datasets = {
        'train': datasets.ImageFolder(root=dirs['train'], transform=data_transforms),
        'test': datasets.ImageFolder(root=dirs['test'], transform=data_size_transforms),
        'valid': datasets.ImageFolder(root=dirs['valid'], transform=data_size_transforms)
    }

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=64, shuffle=True)
                  for x in ['train', 'valid','test']}

    return dataloaders, image_datasets

def create_sequence(features, hidden_units, num_labels,drop_out):
    '''
        creates the classifier for the model
    '''
#     hidden_layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
#     layers      = [nn.Linear(h1, h2) for h1, h2 in hidden_layer_sizes]
#     modules     = list(chain.from_iterable([[layer,nn.ReLU(),nn.Dropout(p=drop_p)]  for layer in layers]))[:-2]
#     modules.append(nn.LogSoftmax(dim=1))    
#     return modules

    return torch.nn.Sequential(OrderedDict([
        ('fc1', torch.nn.Linear(features, 4096)),
        ('norm', nn.BatchNorm1d(4096)),
        ('relu', torch.nn.ReLU()),
        ('dropout', torch.nn.Dropout(p=drop_out)),
        ('output', torch.nn.Linear(4096, num_labels)),
        ('softmax',nn.LogSoftmax(dim=1))
    ]))
    
def get_model(in_args, num_labels):
    '''
        Create the model, using a base architecture.
    '''
    arch    = in_args.arch
    hidden_layers = in_args.hidden_units
    
    print("in_args : ",in_args)
    print("hidden_layers : ",hidden_layers)
    
    if arch in ["vgg19", "densenet161"]:
        print("arch : ", arch)
        if arch == "vgg19":
            model = models.vgg19(pretrained=True) 
            num_in_features = model.classifier[0].in_features
        if arch == "densenet161":
            model = models.densenet161(pretrained=True)
            num_in_features = 2208
            
    features = create_sequence(num_in_features, in_args.hidden_units, num_labels, 0.15)

    print("features : ",features)
    
    for param in model.parameters():
        param.requires_grad = False

    model.classifier = features
    
    cuda = in_args.gpu and torch.cuda.is_available()

    if cuda:
        model.cuda()
    else:
        model.cpu()
        
    return model

# Implement a function for the validation pass
def validation(in_args, model, testloader, criterion):
    '''
        Validate the models preformance against the validation dataset.
    '''
    test_loss = 0
    accuracy = 0
    cuda = in_args.gpu and torch.cuda.is_available()
    
    for images, labels in testloader:
        if cuda == True:
            inputs, labels = Variable(images.cuda()), Variable(labels.cuda())
        else:
            inputs, labels = Variable(images), Variable(labels)

        output = model.forward(inputs)
        test_loss += criterion(output, labels).item()

        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.cuda.FloatTensor).mean()
    
    return test_loss, accuracy
    
def train_model(in_args, dataloaders, image_datasets):
    '''
        The main training step, this is based off what was learnt during the lectures. 
    '''
    epochs = in_args.epochs 
    learning_rate = in_args.learning_rate
    print_every = 40
    steps = 0
    running_loss = 0
    
    num_labels = len(image_datasets['train'].classes)
    model = get_model(in_args, num_labels)
    
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    cuda = in_args.gpu and torch.cuda.is_available()
    
    for e in range(epochs):
        model.train()
        running_loss = 0
        for images, labels in dataloaders['train']:
            if cuda == True:
                inputs, labels = Variable(images.cuda()), Variable(labels.cuda())
            else:
                inputs, labels = Variable(images), Variable(labels)
                
            steps += 1
            optimizer.zero_grad()         
            output = model.forward(inputs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                # Make sure network is in eval mode for inference
                model.eval()

                # Turn off gradients for validation, saves memory and computations
                with torch.no_grad():
                    test_loss, accuracy = validation(in_args, model, dataloaders['valid'], criterion)

                print("Epoch: {}/{}.. ".format(e+1, epochs),
                      "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                      "Test Loss: {:.3f}.. ".format(test_loss/len(dataloaders['valid'])),
                      "Test Accuracy: {:.3f}".format(accuracy/len(dataloaders['valid'])))

                running_loss = 0

                #Make sure training is back on
                model.train()
            
    return model, optimizer            
            
def save_model(in_args, image_datasets, model, optimizer):
    if in_args.checkpoint:
        print ('Saving checkpoint to:', in_args.checkpoint) 
        
        model.class_to_idx = image_datasets['train'].class_to_idx

        # some variables are stored while they might not be used in the prediction step, this creates a bit of redudency.
        checkpoint = {'arch': in_args.arch,
                      'input_size': 2208, # not used when loading model
                      'output_size': 102,  # not used when loading model, and will change based on input json 
                      'batch_size':64,     # not used when loading model
                      'epochs': in_args.epochs,
                      'learning_rate': in_args.learning_rate,
                      'classifier' : model.classifier, 
                      "optimizer": optimizer, # not used when loading model
                      'state_dict': model.state_dict(),
                      'hidden_units': in_args.hidden_units, # not used when loading model
                      'class_to_idx': model.class_to_idx}

        location = in_args.save_dir
        if location[-4:] != '.pth':
            if location[-1:] != '/':
                location+='/'
            else:
                location+='/checkpoint.pth'
        
        torch.save(checkpoint, location)
    
# Main program function defined below
def main():
    start_time = time()
    
    in_args = get_input_args()
    dirs = get_dirs(in_args)
    dataloaders, image_datasets = get_dataloaders(dirs)
    model, optimizer = train_model(in_args, dataloaders, image_datasets)
    save_model(in_args, image_datasets, model, optimizer)
    
    end_time = time()
    tot_time = end_time - start_time
    print("\n** Total Elapsed Runtime:",
          str(int((tot_time/3600)))+":"+str(int((tot_time%3600)/60))+":"
          +str(int((tot_time%3600)%60)) )    
    

# Call to main function to run the program
if __name__ == "__main__":
    main()
    
    
