import _init_paths
from datasets.factory import get_imdb
import cv2
import numpy as np
import visdom

def vis_detections(im, class_name, dets, thresh=0.8):
    """Visual debugging of detections."""
    for i in range(np.minimum(10, dets.shape[0])):
        bbox = tuple(int(np.round(x)) for x in dets[i, :4])
        score = dets[i, -1]
        if score > thresh:
            cv2.rectangle(im, bbox[0:2], bbox[2:4], (0, 204, 0), 2)
            cv2.putText(im, '%s: %.3f' % (class_name, score), (bbox[0], bbox[1] + 15), cv2.FONT_HERSHEY_PLAIN,
                        1.0, (0, 0, 255), thickness=1)
    return im

imdb = get_imdb('voc_2007_trainval')
img_name = imdb.image_path_at(2018)
gt_box = imdb.gt_roidb()[2018]

img = cv2.imread(img_name)
res = vis_detections(img, 'car', gt_box['boxes'])

res = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)

output = res.transpose((2,0,1))
#print(trans.shape)


# output = np.dstack([trans[2,:,:], trans[1,:,:],trans[0,:,:]])
# output = output.transpose((2,0,1))

vis = visdom.Visdom(server='http://localhost',port='8097')
vis.text('Ground Truth of Image')
vis.image(output)
