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
parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=1e-2, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=0.0001, type=float,
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
    model = RNN(num_classes=51, input_size=512, hidden_size=512, batch_size=args.batch_size, num_layers=2, use_gpu=True).cuda()

    model = torch.nn.DataParallel(model)


    print("model loaded")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), 1e-4)

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

    log = Logger(os.path.join(os.getcwd(), 'log'), 'rnn_3')
    print("start training!")

    for epoch in range(args.start_epoch, args.epochs):
        print("Epoch %d", epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, log)

        # evaluate on validation set
        if epoch%args.eval_freq == 0 or epoch == args.epochs-1:
            acc, over = validate(val_loader, model, criterion, epoch, log)
            print("current accuracy: {}".format(acc))
            if over is True:
                break

    print("Loading test set!")
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    test(test_loader, model)


def train(train_loader, model, criterion, optimizer, epoch, log):
    model.train()
    train_loss = AverageMeter()
    for i, (features, target) in enumerate(train_loader):
        batch_size = features.size(0)
        model.batch_size = batch_size

        _, target = torch.max(target, 2)
        target = target.type(torch.LongTensor).cuda(async=True)
        target = target.view(target.size()[0])
        target_var = torch.autograd.Variable(target)
        input_var = torch.autograd.Variable(features, requires_grad=True).cuda()

        output = model(input_var)
        output = torch.mean(output, 1)

        loss = criterion(output, target_var)
        train_loss.update(loss)
        #loss = torch.mean(-torch.log(torch.sum(output*target_var, 1)))


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    log.scalar_summary('training_loss', train_loss.avg, epoch)


def validate(val_loader, model, criterion, epoch, log):
    model.eval()
    over = False
    prec = AverageMeter()
    val_loss = AverageMeter()

    for i, (features, target) in enumerate(val_loader):
        batch_size = features.size(0)
        model.batch_size = batch_size

        _, target = torch.max(target, 2)
        target = target.type(torch.LongTensor).cuda(async=True)
        target = target.view(target.size()[0])
        target_var = torch.autograd.Variable(target, volatile=True)

        input_var = torch.autograd.Variable(features, volatile=True)

        output = model(input_var)
        output = torch.mean(output, 1)

        loss = criterion(output, target_var)
        #loss = torch.mean(-torch.log(torch.sum(output * target_var, 1)))
        prec.update(precision(output.data, target_var.data))
        val_loss.update(loss)

    log.scalar_summary('validate_loss', val_loss.avg, epoch)
    log.scalar_summary('precision', prec.avg, epoch)
    if prec.avg > 0.15:
        over = True
        print("Reach 15%!: ", prec.avg)

    return prec.avg, over


def test(test_loader, model):
    model.eval()
    test_res = open("result/task2_res_adam.txt", "w")

    for i, (features) in enumerate(test_loader):
        batch_size = features.size(0)
        model.batch_size = batch_size

        input_var = torch.autograd.Variable(features, volatile=True)

        output = model(input_var)
        output = torch.mean(output, 1)

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
    gt_cls = gt
    #gt_cls = np.argmax(gt, axis=1)
    pred_cls = np.argmax(pred, axis=1)
    return sklearn.metrics.precision_score(gt_cls, pred_cls, average='macro')


if __name__ == '__main__':
    main()