import sklearn.metrics
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

def compute_map(gt, pred, valid, average=None):
    """
    Compute the multi-label classification accuracy.
    gt (np.ndarray): Shape Nx20, 0 or 1, 1 if the object i is present in that
        image.
    pred (np.ndarray): Shape Nx20, probability of that object in the image
        (output probablitiy).
    valid (np.ndarray): Shape Nx20, 0 if you want to ignore that class for that
        image. Some objects are labeled as ambiguous.
    """
    nclasses = gt.shape[1]
    all_ap = []
    for cid in range(nclasses):
        gt_cls = gt[:, cid][valid[:, cid] > 0].astype('float32')
        pred_cls = pred[:, cid][valid[:, cid] > 0].astype('float32')
        # As per PhilK. code:
        # https://github.com/philkr/voc-classification/blob/master/src/train_cls.py
        pred_cls -= 1e-5 * gt_cls
        ap = sklearn.metrics.average_precision_score(
            gt_cls, pred_cls, average=average)
        all_ap.append(ap)
        precision, recall, _ = precision_recall_curve(gt_cls, pred_cls)

        plt.step(recall, precision, color='b', alpha=0.2,
                 where='post')
        plt.fill_between(recall, precision, step='post', alpha=0.2,
                         color='b')
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title('2-class Precision-Recall curve')
        plt.show()
        plt.waitforbuttonpress()
    return all_ap