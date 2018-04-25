import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys
def parse_args():
    parser = argparse.ArgumentParser(
        description='Plot mAP!')
    parser.add_argument(
        'mAP_file', type=str,
        help='Path to mAP file')
    parser.add_argument(
        'max_iter', type=int,
        help='iteration number')
    parser.add_argument(
        'output_file', type=str,
        help='Path to output image')
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    iter_num = args.max_iter
    file_name = args.mAP_file
    total_file = open(file_name).readlines()
    mAP_list = []

    for i in range(len(total_file)):
        mAP_list.append(float(total_file[i].split()[0]))
	interval = iter_num/len(total_file)
    x_iter = np.arange(interval, iter_num + interval, interval)
    # plt.plot(x_iter, loss_list, 'b')
    # plt.xlabel('iterations')
    # plt.ylabel('loss')
    plt.plot(x_iter, mAP_list, 'g')
    plt.xlabel('iterations')
    plt.ylabel('mAP')

    out_file = args.output_file + ".png"
    plt.savefig(out_file)	

if __name__ == "__main__":
    main() 