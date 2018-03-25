
# coding: utf-8

# ### Sample training script annotated with explanations.

# ##### Constant definitions

# In[42]:

import os
import shutil
import time

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn

TRAIN_DATA_DIR = './data/face_10/train'
VAL_DATA_DIR = './data/face_10/val'
TEST_DATA_DIR = './data/face_10/test'
SUBMISSIONS_DIR = './submissions/'
 
INPUT_SIZE = 224  # Alexnet input is 224
BATCH_SIZE = 64  #  Mini batch size
NUM_CLASSES = 10 # Number of classes for our problem

LEARNING_RATE = 1e-2 # Base Learning rate
MOMENTUM = 0.9       # SGD momentum
WEIGHT_DECAY = 1e-4  # SGD Weight decay
EPOCHS = 30          # Total number of epochs to run
PRINT_FREQ = 10      # Prints after going through 10 * batch_size samples

RESUME_FROM = ''     # Checkpoint to resume training from.
FACE_PORT = 1050
HOST = 'hvfaceserver-team4'

import numpy as np
import zipfile
import glob

def getNextSubmissionId():
    a = glob.glob(SUBMISSIONS_DIR+'*.zip')
    try:
        last = max([int(x.split('/')[-1].split('.')[0]) for x in a])
    except:
        last = 0
    return str(last + 1)

def create_submission(preds):
    assert preds.dtype=='int'
    temp = open(SUBMISSIONS_DIR+'answer.txt', 'w')
    for a in range(preds.shape[0]):
        print(int(preds[a]), file=temp)
    temp.close()
    zname = SUBMISSIONS_DIR + getNextSubmissionId()+'.zip'
    z = zipfile.ZipFile(zname, 'w')
    z.write(SUBMISSIONS_DIR+'answer.txt', 'answer.txt')
    z.close()


# ##### Loading Dataset
# 
# The following function can be used to load the dataset specified in the folder. This will return two
# data iterators - one for train and one for val.
# 
# You can also see the transforms and weights applied to the samples of data while loading.

# In[52]:

import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms

def get_train_dataloader():
    traindir = TRAIN_DATA_DIR

    # Transformations applied to the input data
    # while loading.
    
    # Subtract the mean and divide by variance for each RGB Value
    # in the batch.
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224), # Taking a random crop of size 224 x 224.
            transforms.ToTensor(),
            normalize,
        ]))
    
    # Weights governing how likely is one sample over another.
    # Check 
    # - http://pytorch.org/docs/0.3.1/_modules/torch/utils/data/sampler.html#WeightedRandomSampler
    # - https://discuss.pytorch.org/t/how-to-prevent-overfitting/1902/25
    # for more details.
    # Default: Equal weights for all samples.
    weights = torch.ones(len(train_dataset)).double()
    train_sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(train_dataset))
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=(train_sampler is None),
        num_workers=4, 
        pin_memory=True, 
        sampler=train_sampler)
    
    return train_loader


# In[54]:

train_loader = get_train_dataloader()
dataset = train_loader.dataset
print(dataset.transform)
print(dataset.classes)
a=next(iter(train_loader))
print(a[0].size(),a[1])  #batch size


# ##### Loading Alexnet architecture
# 
# You will find the Alexnet architecture which was discussed in the presentation
# defined below using `torch.nn` and `torch.nn.Module` modules
# 
# The pretrained weights are copied into the model wherever the parameter names 
# and sizes match. Else, It is randomly initialized using Xavier Init.

# In[55]:

import torch.utils.model_zoo as model_zoo
import torch.nn as nn

class AlexNet(nn.Module):

    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x


def alexnet(pretrained=False, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    url = 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth'
    model = AlexNet(**kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(url))
    return model

def load_alexnet(num_classes, pretrained = True):
    
    model = alexnet(pretrained=False, num_classes=num_classes)
    alexnet_url = 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth'
    
    if pretrained:
      
        print("=> using pre-trained model '{}'".format('alexnet'))
        print()
        
        pretrained_state = model_zoo.load_url(alexnet_url)
        model_state = model.state_dict()

        unfreeze = [ k for k in model_state 
                        if k not in pretrained_state 
                        or pretrained_state[k].size() != model_state[k].size() ]
        
        ignored_states = ','.join([x for x in pretrained_state 
                                       if x not in model_state])
        
        print("=" * 80)
        print("--> Ignoring '{}' during restore".format(ignored_states))
        print("=" * 80)
        print("--> '{}' - Cannot copy parameters due to size mismatch / not present "
              "in pretrained model. Init with random".format(','.join([x for x in unfreeze])))
        print("=" * 80)
        
        pretrained_state = { k:v for k,v in pretrained_state.items() 
                if k in model_state and v.size() == model_state[k].size() }
        
        model_state.update(pretrained_state)
        model.load_state_dict(model_state)
        
    return model


# In[48]:

model=load_alexnet(10)


# In[56]:

for p in model.named_parameters():
    print(p[0],p[1].data.size())


# ##### Utilities
# 
# For computing moving averages, accuracy and saving the Checkpoints.

# In[36]:

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def save_checkpoint(state, is_best, filename):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, './models/model_best_' + str(NUM_CLASSES) + '_class.pth.tar')


# ##### Training loop
# 
# The train function does the following
# - Loops over the data in an epoch using the data loader
# - Applies the model on the input to get the output.
# - Computes the loss for each iteration.
# - Computes the gradient of loss w.r.t model parameters
# - Updates the weights.

# In[60]:

def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        input = input.cuda()
        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % PRINT_FREQ == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch + 1, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))


# ##### Validation Loop
# 
# The `validate` function does the following
# - Iterates over the validation data using the val_loader
# - Computes the predictions for the validation samples.
# - Computes the validation loss and validation accuracy.
# - Returns the validation accuracy.

# In[61]:

def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        
        input = input.cuda()
        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % PRINT_FREQ == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses,
                   top1=top1, top5=top5))

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg


# ##### Predictions
# 
# We need to use our best model to submit predictions on the test data for a given task. The following function does that for you on a batch of test data and returns predictions as a numpy array.
# 
# - The function accepts the checkpoint path. It will load the model based on the Model architecture defined and copies the weights from the checkpoint file.
# - The function returns the class predictions for the data in the test dir.

# In[64]:

def predict(model_path):
    
     # Load Model from checkpoint
    if(model_path is None):
        print('checkpoint argument cannot be None')
        return None
    
    if os.path.isfile(model_path) == False:
        print('{} is not found'.format(model_path))
        return None
    
    model = load_alexnet(num_classes = NUM_CLASSES, pretrained=False)
    
    print("=> loading checkpoint '{}'".format(model_path))
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])
    print("=> loaded model from checkpoint '{}'".format(model_path))

    is_cuda_available = torch.cuda.is_available()
    
    if is_cuda_available:
        model.cuda()
        
    # test data loader - Keep it similar to val data loader
    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    
    test_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(TEST_DATA_DIR, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=BATCH_SIZE, 
        shuffle=False,
        num_workers=4, 
        pin_memory=True)
    
   
    num_batches = len(test_loader)
    num_elements = len(test_loader.dataset)
    batch_size = test_loader.batch_size

    pred_array = torch.zeros(num_elements).long()
    prob_array = torch.zeros(num_elements, NUM_CLASSES)

    model.train(False)

    
    for i, data in enumerate(test_loader):
        
        # Get the indices for each batch
        start = i*batch_size
        end = start + batch_size
        if i == num_batches - 1:
            end = num_elements
        
        inputs, _ = data
        
        # wrap them in Variable
        if is_cuda_available:
            inputs = torch.autograd.Variable(inputs.cuda(), volatile=True)
        else:
            inputs = torch.autograd.Variable(inputs)
        
        # forward
        outputs = model(inputs)
        _, preds = torch.max(outputs.data, 1)
        
        # compute output
        
        pred_array[start:end], prob_array[start:end] = preds.long(), outputs.data

    return pred_array.numpy(), prob_array.numpy()

def make_submisison_for_model(model_path = './models/model_best_' + str(NUM_CLASSES) + '_class.pth.tar'):
    predictions, probabilities = predict(model_path)
    create_submission(predictions)


# In[2]:

import json
import urllib
from urllib import request
import socket
from socket import *
from PIL import Image

def get_prediction_for_url(image_url, 
                           img_transform, 
                           idx_to_class, 
                           model_path='./models/model_best_' + str(NUM_CLASSES) + '_class.pth.tar'):
    
    # Load Model from checkpoint
    if(model_path is None):
        print('checkpoint argument cannot be None')
        return None
    
    if os.path.isfile(model_path) == False:
        print('{} is not found'.format(model_path))
        return None
    
    model = load_alexnet(num_classes = NUM_CLASSES, pretrained=False)
    
    print("=> loading checkpoint '{}'".format(model_path))
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])
    print("=> loaded model from checkpoint '{}'".format(model_path))

    is_cuda_available = torch.cuda.is_available()
    
    if is_cuda_available:
        model.cuda()

    # Download the image from url and save it to disk
    f = open('/scratch/input.jpg', 'wb')
    f.write(request.urlopen(image_url).read())
    f.close()
    
    # Detect and get aligned face from hvfaceserver
    requestJson = {}
    requestJson['method'] = '/face/align'
    requestJson['image'] = '/scratch/input.jpg'
    requestJson['output'] = '/scratch/output.jpg'

    # Send a request to the HV Face server
    s = socket(AF_INET, SOCK_STREAM)
    s.connect((HOST, FACE_PORT))
    s.send(json.dumps(requestJson).encode())
    recdata = s.recv(200000)
    s.close()
    
    if not recdata:
        print('No response received from HV Face Server')
        return None
    
    response = json.loads(recdata)
        
    if 'error' in response:
        print('Error occurred in getting response from HV Face Server for url - {}'.format(response['error']))
        return None
        
    output_image = response['output']
    
    with open(output_image, 'rb') as f:
        img = Image.open(f).convert('RGB')      
        
    img_tensor = img_transform(img).unsqueeze(0)
    
    if is_cuda_available:
        img_tensor = img_tensor.cuda()
    
    inputs = torch.autograd.Variable(img_tensor, volatile=True)
    outputs = model(inputs)
    _, preds = torch.max(outputs.data, 1)
    
    return idx_to_class[preds.cpu().numpy()[0]]
    
    


# ##### Putting it all together
# 
# - Load the datasets and get the dataloaders
# - Load the model
# - Decide the loss criterion
# - Decide the optimizer
# - Decide the Learning rate schedule.
# - Train and Validate.
# - Use best model to predict on unseen data

# In[15]:

def main():
    
    # Load the train loader. Have a look at the function.
    train_loader = get_train_dataloader()
   
    # Here is an example of creating own loader. Useful for loading val
    # and test datasets which have similar transforms.
    
    # define the data directory
    val_dir = VAL_DATA_DIR
    
    # define all the transforms.
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    val_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])
    
    # get the loader
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(val_dir, val_transforms),
        batch_size=BATCH_SIZE, 
        shuffle=False,  # Shuffling not necessary for val and test.
        num_workers=4,  
        pin_memory=True)
    
 
    # Load the pretrained alexnet model repurposed for our NUM_CLASSES
    model = load_alexnet(num_classes = NUM_CLASSES, pretrained = True)
    
    # Using the CrossEntropyLoss
    criterion = nn.CrossEntropyLoss()

    # Check if GPU is available.
    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()

    # Setup the optimizer to track the model parameters to update.
    optimizer = torch.optim.SGD(model.parameters(), 
                                LEARNING_RATE,
                                momentum=MOMENTUM,
                                weight_decay=WEIGHT_DECAY)
    
    
    start_epoch = 0
    best_prec1 = 0;
    
    # Resume from checkpoint. Useful if you are pausing the training to
    # change certain hyperparameters ( data, base learning rate)
    
    if RESUME_FROM:
        if os.path.isfile(RESUME_FROM):
            print("=> loading checkpoint '{}'".format(RESUME_FROM))
            checkpoint = torch.load(RESUME_FROM)
            start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(RESUME_FROM, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(RESUME_FROM))
    
    # setting up the learning rate schedule - how it should change
    # as learning progresses. Check out other schedulers on the PyTorch
    # Documentation website
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
                                    step_size=15, 
                                    gamma=0.1, 
                                    last_epoch=(start_epoch - 1))
    
    # Lets roll!
    for epoch in range(start_epoch, EPOCHS):
        # Call the learning rate scheduler every epoch to
        # update learning rate if necessary.
        scheduler.step()

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion)

        # remember best prec@1 and save checkpoint
        # if it is atleast 80%
        
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        if best_prec1 > 70:
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': 'alexnet',
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'optimizer' : optimizer.state_dict(),
            }, is_best, filename = './models/checkpoint_' + str(NUM_CLASSES) + '_class.pth.tar')


# In[16]:

main()


# In[24]:

# define all the transforms.
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
img_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

# get the idx_to_class mapping
val_dir = VAL_DATA_DIR
valset = datasets.ImageFolder(val_dir, None)
idx_to_class = {v:k for k,v in valset.class_to_idx.items()}

get_prediction_for_url('https://www.hindustantimes.com/rf/image_size_960x540/HT/p2/2017/04/22/Pictures/rajinikanth-2-o_7b65b59c-271c-11e7-b743-a11580b053fc.jpg', 
                       img_transform,
                       idx_to_class,
                       model_path='./models/model_best_' + str(NUM_CLASSES) + '_class.pth.tar')


# In[18]:

make_submisison_for_model('./models/model_best_100_class.pth.tar')


# In[ ]:



