import argparse
import os
import shutil
import time
import sys
sys.path.insert(0,'/home/spurushw/reps/hw-wsddn-sol/faster_rcnn')
import sklearn
import sklearn.metrics

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

import matplotlib.pyplot as plt
import cv2
import visdom
import _init_paths
from datasets.factory import get_imdb
from custom import *
from logger import *



model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--arch', default='localizer_alexnet')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=2, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--eval-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='gloo', type=str,
                    help='distributed backend')
parser.add_argument('--vis',action='store_true')

best_prec1 = 0

classes_name = ('aeroplane', 'bicycle', 'bird', 'boat',
                 'bottle', 'bus', 'car', 'cat', 'chair',
                 'cow', 'diningtable', 'dog', 'horse',
                 'motorbike', 'person', 'pottedplant',
                 'sheep', 'sofa', 'train', 'tvmonitor')


def main():
    global args, best_prec1
    args = parser.parse_args()
    args.distributed = args.world_size > 1
    vis = visdom.Visdom(server='http://localhost',port='8097')


    # create model
    print("=> creating model '{}'".format(args.arch))
    if args.arch=='localizer_alexnet':
        model = localizer_alexnet(pretrained=args.pretrained)
    elif args.arch=='localizer_alexnet_robust':
        model = localizer_alexnet_robust(pretrained=args.pretrained)
    print(model)

    model.features = torch.nn.DataParallel(model.features)
    model.cuda()

    # TODO:
    # define loss function (criterion) and optimizer

    criterion = nn.BCELoss().cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr, 
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    print("Loss function and optimizer decided!")

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    # TODO: Write code for IMDBDataset in custom.py
    trainval_imdb = get_imdb('voc_2007_trainval')
    test_imdb = get_imdb('voc_2007_test')

    print("Data loading!")

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    print("Loading training set!")
    train_dataset = IMDBDataset(
        trainval_imdb,
        transforms.Compose([
            transforms.Scale((512,512)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, worker_init_fn=0)

    print("Loading test set!")
    val_loader = torch.utils.data.DataLoader(
        IMDBDataset(test_imdb, transforms.Compose([
            transforms.Scale((384,384)),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, worker_init_fn=0)

    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    # TODO: Create loggers for visdom and tboard
    # TODO: You can pass the logger objects to train(), make appropriate
    # modifications to train()

    log = Logger(os.path.join(os.getcwd(), 'log'), 'freeloc_test')
    

    print("start training!")


    for epoch in range(args.start_epoch, args.epochs):
        print("Epoch %d", epoch)

        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, log, vis)

        # evaluate on validation set
        if epoch%args.eval_freq==0 or epoch==args.epochs-1:
            m1, m2 = validate(val_loader, model, criterion, epoch, log, vis)
            score = m1*m2
            # remember best prec@1 and save checkpoint
            is_best =  score > best_prec1
            best_prec1 = max(score, best_prec1)
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'optimizer' : optimizer.state_dict(),
            }, is_best)


def inv_normalize(y, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    # Function for inverse normalization
    if isinstance(y, torch.Tensor) :
        x = y.new(*y.size())
    else:
        x = np.zeros(y.shape)
        
    x[0, :, :] = y[0, :, :] * std[0] + mean[0]
    x[1, :, :] = y[1, :, :] * std[1] + mean[1]
    x[2, :, :] = y[2, :, :] * std[2] + mean[2]
    return x


#TODO: You can add input arguments if you wish
def train(train_loader, model, criterion, optimizer, epoch, log, vis):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    avg_m1 = AverageMeter()
    avg_m2 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()


    episode = len(train_loader)
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.type(torch.FloatTensor).cuda(async=True)
        input_var = torch.autograd.Variable(input, requires_grad=True)
        target_var = torch.autograd.Variable(target)

        # TODO: Get output from model
        # TODO: Perform any necessary functions on the output
        # TODO: Compute loss using ``criterion``
        # compute output
        output = model(input_var)

        # Global pooling and compute Sigmoid
        pool_res = F.sigmoid(F.max_pool2d(output, output.size()[2]))
        pool_res = pool_res.view(pool_res.size()[0], -1)

        if i == 0 and epoch % 15 == 0:
            all_gt_classes = target.cpu().numpy()
            input_images = input.cpu().numpy()

            # Upsampling output to similar size of input image for heatmap
            upsample = nn.Upsample(scale_factor=17, mode='bilinear', align_corners=True)
            heat_maps = upsample(output).data.cpu().numpy()

            log.image_summary('{} epoch input image'.format(epoch), input_images, epoch)
            
            img_num = input_images.shape[0]
            for k in range(img_num):
                gt_classes = all_gt_classes[k]
                input_image = input_images[k]
                input_image = inv_normalize(input_image)
                
                #vis.text('{} Input image'.format(episode * epoch + i))
                vis.image(input_image,
                    opts=dict(title='Train input epoch_{}_{}'.format(epoch, k), 
                        caption='{} iteration input image'.format(episode * epoch + i)))
                #print(gt_classes)
                #vis.text('{} HeatMap'.format(episode * epoch + i))
                heat_map = heat_maps[k]
                
                vis.heatmap(X=heat_map[gt_classes.argmax()],
                    opts=dict(title='Train HeatMap_' + classes_name[gt_classes.argmax()] + ' {}_{}'.format(epoch, k),
                        caption='{} iteration heatmap'.format(episode * epoch + i)))

                #print(heat_map[gt_classes.argmax()].shape)
                cm = plt.get_cmap('plasma')
                colored_heatmap = cm(heat_map[gt_classes.argmax()])
                res_heatmap = (colored_heatmap[:, :, :3] * 255).astype(np.uint8).reshape((1, 29, 29, 3))
                log.image_summary('{} epoch {}th heatmap'.format(epoch, k), res_heatmap, epoch)

        # print("Output size: ",output.size())
        # print("Target size: ", target_var.size())

        loss = criterion(pool_res, target_var)
        

        # measure metrics and record loss
        m1 = metric1(pool_res.data, target)
        m2 = metric2(pool_res.data, target)
        losses.update(loss.data[0], input.size(0))
        avg_m1.update(m1[0], input.size(0))
        avg_m2.update(m2[0], input.size(0))

        global_step = episode * epoch + i
        log.scalar_summary('training_loss', loss, global_step)
        log.scalar_summary('train_metric1', m1[0], global_step)
        log.scalar_summary('train_metric2', m2[0], global_step)
        
        # TODO: 
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()




        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Metric1 {avg_m1.val:.3f} ({avg_m1.avg:.3f})\t'
                  'Metric2 {avg_m2.val:.3f} ({avg_m2.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, avg_m1=avg_m1,
                   avg_m2=avg_m2))

        #TODO: Visualize things as mentioned in handout
        #TODO: Visualize at appropriate intervals
        

def validate(val_loader, model, criterion, epoch, log, vis):
    batch_time = AverageMeter()
    losses = AverageMeter()
    avg_m1 = AverageMeter()
    avg_m2 = AverageMeter()

    episode = len(val_loader)

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.type(torch.FloatTensor).cuda(async=True)
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # TODO: Get output from model
        # TODO: Perform any necessary functions on the output
        # TODO: Compute loss using ``criterion``
        # compute output
        output = model(input_var)
        pool_res = F.sigmoid(F.max_pool2d(output, output.size()[2]))
        pool_res = pool_res.view(pool_res.size()[0], -1)

        if i == 0 and epoch % 15 == 0:
            all_gt_classes = target.cpu().numpy()
            input_images = input.cpu().numpy()
            heat_maps = output.data.cpu().numpy()

            log.image_summary('{} epoch input image'.format(epoch), input_images, epoch)
            
            img_num = input_images.shape[0]
            for k in range(img_num):
                gt_classes = all_gt_classes[k]
                input_image = input_images[k]
                input_image = inv_normalize(input_image)
                
                #vis.text('{} Input image'.format(episode * epoch + i))
                vis.image(input_image,
                    opts=dict(title='validate input epoch_{}_{}'.format(epoch, k), 
                        caption='{} iteration input image'.format(episode * epoch + i)))
                #print(gt_classes)
                #vis.text('{} HeatMap'.format(episode * epoch + i))
                heat_map = heat_maps[k]
                
                vis.heatmap(X=heat_map[gt_classes.argmax()],
                    opts=dict(title='validate HeatMap_' + classes_name[gt_classes.argmax()] + ' {}_{}'.format(epoch, k),
                        caption='{} iteration heatmap'.format(episode * epoch + i)))

                #print(heat_map[gt_classes.argmax()].shape)
                #log.image_summary('{} epoch {}th heatmap'.format(epoch, k), heat_map[gt_classes.argmax()], epoch * 32 + k)


        loss = criterion(pool_res, target_var)


        # measure metrics and record loss
        m1 = metric1(pool_res.data, target)
        m2 = metric2(pool_res.data, target)
        losses.update(loss.data[0], input.size(0))
        avg_m1.update(m1[0], input.size(0))
        avg_m2.update(m2[0], input.size(0))

        global_step = episode * epoch + i
        log.scalar_summary('validate_loss', loss, global_step)
        log.scalar_summary('validate_metric1', m1[0], global_step)
        log.scalar_summary('validate_metric2', m2[0], global_step)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Metric1 {avg_m1.val:.3f} ({avg_m1.avg:.3f})\t'
                  'Metric2 {avg_m2.val:.3f} ({avg_m2.avg:.3f})'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses,
                   avg_m1=avg_m1, avg_m2=avg_m2))

        #TODO: Visualize things as mentioned in handout
        #TODO: Visualize at appropriate intervals





    print(' * Metric1 {avg_m1.avg:.3f} Metric2 {avg_m2.avg:.3f}'
          .format(avg_m1=avg_m1, avg_m2=avg_m2))

    return avg_m1.avg, avg_m2.avg


# TODO: You can make changes to this function if you wish (not necessary)
def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


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


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def metric1(output, target):
    # Calculate mAP 
    gt = target.cpu().numpy()
    pred = output.cpu().numpy()
    gt_cls = gt.astype('float32')
    pred_cls = pred.astype('float32')
    pred_cls -= 1e-5 * gt_cls
    mAP = 0
    for i in range(gt_cls.shape[0]):
        mAP += sklearn.metrics.average_precision_score(gt_cls[i], pred_cls[i])

    return [mAP/gt_cls.shape[0]]

def metric2(output, target):
    # TODO: Ignore for now - proceed till instructed
    return [0]

if __name__ == '__main__':
    main()
