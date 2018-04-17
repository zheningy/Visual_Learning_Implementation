import argparse
import sklearn
import sklearn.metrics

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed


import pickle
import random
from network import *
from logger import *



parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--arch', default='localizer_alexnet')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=1e-2, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=0.01, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--eval-freq', default=2, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--vis',action='store_true')


def get_data(file, shuffle=False):
    data_file = open(file, 'rb')
    dataset = pickle.load(data_file)
    dataset = dataset['data']
    if shuffle:
        random.shuffle(dataset)
    return dataset


def main():
    global args
    args = parser.parse_args()
    model = SimpleNet()
    model.classifer = torch.nn.DataParallel(model.classifer)
    model.cuda()

    print("model loaded")

    criterion = nn.BCELoss().cuda()
    # optimizer = torch.optim.Adam(model.parameters(), args.lr)
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    print("Loss function and optimizer decided!")

    trainval_data = get_data('annotated_train_set.p', shuffle=True)
    test_data = get_data('randomized_annotated_test_set_no_name_no_num.p')
    num_samples = len(trainval_data)
    k_fold = 4
    num_train = int((1-1.0/k_fold) * num_samples)
    train_dataset = HDMBDataset(trainval_data[:num_train])
    val_dataset = HDMBDataset(trainval_data[num_train:])
    test_dataset = HDMBDataset(test_data)

    print("Loading training set!")
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=False,#(train_sampler is None),
        num_workers=args.workers, pin_memory=True)

    print("Loading validation set!")
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,#(train_sampler is None),
        num_workers=args.workers, pin_memory=True)

    log = Logger(os.path.join(os.getcwd(), 'log'), 'sheep_net')
    print("start training!")

    for epoch in range(args.start_epoch, args.epochs):
        print("Epoch %d", epoch)

        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, log)

        # evaluate on validation set
        if epoch%args.eval_freq == 0 or epoch == args.epochs-1:
            acc = validate(val_loader, model, criterion, epoch, log)
            print("current accuracy: {}".format(acc))

    print("Loading test set!")
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,#(train_sampler is None),
        num_workers=args.workers, pin_memory=True)

    test(test_loader, model)



def train(train_loader, model, criterion, optimizer, epoch, log):
    """
    Train the model
    :param train_loader:
    :param model:
    :param criterion:
    :param optimizer:
    :param epoch:
    :param log:
    :return:
    """
    model.train()
    train_loss = AverageMeter()
    for i, (features, target) in enumerate(train_loader):
        target = target.type(torch.FloatTensor).cuda(async=True)
        target = target.view(-1, target.size()[2])
        target_var = torch.autograd.Variable(target)

        features = features.view(features.size(0), -1)
        input_var = torch.autograd.Variable(features, requires_grad=True)

        output = model(input_var)

        #output = torch.cat([model(torch.autograd.Variable(features[x], requires_grad=True)) for x in range(features.shape[0])], 0)
        #loss = criterion(output, target_var)
        loss = torch.mean(-torch.log(torch.sum(output*target_var, 1)))
        train_loss.update(loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    log.scalar_summary('training_loss', train_loss.avg, epoch)


def validate(val_loader, model, criterion, epoch, log):
    """
    evaluate the model during training
    :param val_loader:
    :param model:
    :param criterion:
    :param epoch:
    :param log:
    :return:
    """
    avg_m1 = AverageMeter()
    prec = AverageMeter()
    val_loss = AverageMeter()
    model.eval()

    for i, (features, target) in enumerate(val_loader):
        target = target.type(torch.FloatTensor).cuda(async=True)
        target = target.view(-1, target.size()[2])
        target_var = torch.autograd.Variable(target, volatile=True)

        features = features.view(features.size(0), -1)
        input_var = torch.autograd.Variable(features, volatile=True)
        output = model(input_var)

        #loss = criterion(output, target_var)
        loss = torch.mean(-torch.log(torch.sum(output * target_var, 1)))
        m1 = metric1(output.data, target_var.data)
        prec.update(precision(output.data, target_var.data))
        val_loss.update(loss)

        avg_m1.update(m1[0], features.size(0))
    log.scalar_summary('validate_loss', val_loss.avg, epoch)
    log.scalar_summary('validate_mAP', avg_m1.avg, epoch)
    log.scalar_summary('precision', prec.avg, epoch)

    return prec.avg


def test(test_loader, model):
    model.eval()
    test_res = open("result/task1_res.txt", "w")

    for i, (features) in enumerate(test_loader):
        features = features.view(features.size(0), -1)
        input_var = torch.autograd.Variable(features, volatile=True)
        output = model(input_var)

        curt_res = output.data.cpu().numpy()
        pred = np.argmax(curt_res, axis=1)
        for i in pred:
            test_res.write(str(i))
            test_res.write('\n')





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

def precision(output, target):
    gt = target.cpu().numpy()
    pred = output.cpu().numpy()
    gt_cls = np.argmax(gt, axis=1)
    pred_cls = np.argmax(pred, axis=1)
    return sklearn.metrics.precision_score(gt_cls, pred_cls, average='macro')



def metric1(output, target):
    # Calculate mAP
    gt = target.cpu().numpy()
    pred = output.cpu().numpy()
    gt_cls = gt.astype('float32')
    pred_cls = pred.astype('float32')
    #pred_cls -= 1e-5 * gt_cls
    mAP = 0
    for i in range(gt_cls.shape[0]):
        mAP += sklearn.metrics.average_precision_score(gt_cls[i], pred_cls[i])

    return [mAP/gt_cls.shape[0]]


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    main()