# -*- coding: utf-8 -*-
# File: eval.py

import tqdm
import os
from collections import namedtuple
import numpy as np
import cv2

from tensorpack import *
from tensorpack.utils.utils import get_tqdm_kwargs

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import pycocotools.mask as cocomask

from coco import COCOMeta
from common import CustomResize, clip_boxes
from config import config as cfg

DetectionResult = namedtuple(
    'DetectionResult',
    ['box', 'score', 'class_id', 'mask'])
"""
box: 4 float
score: float
class_id: int, 1~NUM_CLASS
mask: None, or a binary image of the original image shape
"""


def non_max_suppression_fast(boxes, probs=None, overlapThresh=0.3):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # if the bounding boxes are integers, convert them to floats -- this
    # is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # compute the area of the bounding boxes and grab the indexes to sort
    # (in the case that no probabilities are provided, simply sort on the
    # bottom-left y-coordinate)
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = y2

    # if probabilities are provided, sort on them instead
    if probs is not None:
        idxs = probs

    # sort the indexes
    idxs = np.argsort(idxs)

    # keep looping while some indexes still remain in the indexes list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the index value
        # to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of the bounding
        # box and the smallest (x, y) coordinates for the end of the bounding
        # box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have overlap greater
        # than the provided overlap threshold
        idxs = np.delete(idxs, np.concatenate(([last],
                                               np.where(overlap > overlapThresh)[0])))

    # return only the bounding boxes that were picked
    logger.info('PICK!!')
    logger.info(pick)

    return boxes[pick]



def fill_full_mask(box, mask, shape):
    """
    Args:
        box: 4 float
        mask: MxM floats
        shape: h,w
    """
    # int() is floor
    # box fpcoor=0.0 -> intcoor=0.0
    x0, y0 = list(map(int, box[:2] + 0.5))
    # box fpcoor=h -> intcoor=h-1, inclusive
    x1, y1 = list(map(int, box[2:] - 0.5))    # inclusive
    x1 = max(x0, x1)    # require at least 1x1
    y1 = max(y0, y1)

    w = x1 + 1 - x0
    h = y1 + 1 - y0

    # rounding errors could happen here, because masks were not originally computed for this shape.
    # but it's hard to do better, because the network does not know the "original" scale
    mask = (cv2.resize(mask, (w, h)) > 0.5).astype('uint8')
    ret = np.zeros(shape, dtype='uint8')
    ret[y0:y1 + 1, x0:x1 + 1] = mask
    return ret


def detect_one_image(img, model_func):
    """
    Run detection on one image, using the TF callable.
    This function should handle the preprocessing internally.

    Args:
        img: an image
        model_func: a callable from TF model,
            takes image and returns (boxes, probs, labels, [masks])

    Returns:
        [DetectionResult]
    """

    orig_shape = img.shape[:2]
    resizer = CustomResize(cfg.PREPROC.SHORT_EDGE_SIZE, cfg.PREPROC.MAX_SIZE)
    resized_img = resizer.augment(img)
    scale = np.sqrt(resized_img.shape[0] * 1.0 / img.shape[0] * resized_img.shape[1] / img.shape[1])
    boxes, probs, labels, *masks = model_func(resized_img)
    boxes = boxes / scale
    # boxes are already clipped inside the graph, but after the floating point scaling, this may not be true any more.
    boxes = clip_boxes(boxes, orig_shape)

    if masks:
        # has mask
        full_masks = [fill_full_mask(box, mask, orig_shape)
                      for box, mask in zip(boxes, masks[0])]
        masks = full_masks
    else:
        # fill with none
        masks = [None] * len(boxes)

    results = [DetectionResult(*args) for args in zip(boxes, probs, labels, masks)]
    return results


def detect_one_image_scale(img, model_func):
    """
    Run detection on one image, using the TF callable.
    This function should handle the preprocessing internally.

    Args:
        img: an image
        model_func: a callable from TF model,
            takes image and returns (boxes, probs, labels, [masks])

    Returns:
        [DetectionResult]
    """
    scores_ts = []
    boxes_ts = []
    labels_ts = []
    masks_ts = []

    def add_preds_t(scores_t, boxes_t, labels_t, masks_t):
        scores_ts.append(scores_t)
        boxes_ts.append(boxes_t)
        labels_ts.append(labels_t)
        masks_ts.append(masks_t)

    orig_shape = img.shape[:2]
    for bbox_aug_scale in cfg.TEST.BBOX_AUG_SCALES:
        resizer = CustomResize(bbox_aug_scale, cfg.TEST.BBOX_AUG_MAX_SIZE)
        resized_img = resizer.augment(img)
        scale = np.sqrt(resized_img.shape[0] * 1.0 / img.shape[0] * resized_img.shape[1] / img.shape[1])
        boxes, probs, labels, *masks = model_func(resized_img)
        boxes = boxes / scale
        # boxes are already clipped inside the graph, but after the floating point scaling, this may not be true any more.
        boxes = clip_boxes(boxes, orig_shape)
        add_preds_t(probs, boxes, labels, masks)

    if cfg.TEST.BBOX_AUG_COORD_HEUR == 'UNION':
        boxes_c = np.vstack(boxes_ts)
        scores_c = np.vstack(scores_ts)
        lables_c = np.vstack(labels_ts)
        masks_c = np.vstack(masks_ts)

        # Apply NMS
        logger.info(boxes_c)

        pick = non_max_suppression_fast(boxes_c, scores_c, 0.5)

        logger.info("detect_one_image_scale...NMS...")
        logger.info(pick)

    if masks:
        # has mask
        full_masks = [fill_full_mask(box, mask, orig_shape)
                      for box, mask in zip(boxes_c, masks_c[0])]
        masks = full_masks
    else:
        # fill with none
        masks = [None] * len(boxes_c)

    results = [DetectionResult(*args) for args in zip(boxes_c, scores_c, lables_c, masks)]
    return results


def eval_coco(df, detect_func):
    """
    Args:
        df: a DataFlow which produces (image, image_id)
        detect_func: a callable, takes [image] and returns [DetectionResult]

    Returns:
        list of dict, to be dumped to COCO json format
    """
    df.reset_state()
    all_results = []
    with tqdm.tqdm(total=df.size(), **get_tqdm_kwargs()) as pbar:
        for img, img_id in df.get_data():
            results = detect_func(img)
            for r in results:
                box = r.box
                cat_id = COCOMeta.class_id_to_category_id[r.class_id]
                box[2] -= box[0]
                box[3] -= box[1]

                res = {
                    'image_id': img_id,
                    'category_id': cat_id,
                    'bbox': list(map(lambda x: round(float(x), 2), box)),
                    'score': round(float(r.score), 3),
                }

                # also append segmentation to results
                if r.mask is not None:
                    rle = cocomask.encode(
                        np.array(r.mask[:, :, None], order='F'))[0]
                    rle['counts'] = rle['counts'].decode('ascii')
                    res['segmentation'] = rle
                all_results.append(res)
            pbar.update(1)
    return all_results


# https://github.com/pdollar/coco/blob/master/PythonAPI/pycocoEvalDemo.ipynb
def print_evaluation_scores(json_file):
    ret = {}
    assert cfg.DATA.BASEDIR and os.path.isdir(cfg.DATA.BASEDIR)
    annofile = os.path.join(
        cfg.DATA.BASEDIR, 'annotations',
        'instances_{}.json'.format(cfg.DATA.VAL))
    coco = COCO(annofile)
    cocoDt = coco.loadRes(json_file)
    cocoEval = COCOeval(coco, cocoDt, 'bbox')
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
    fields = ['IoU=0.5:0.95', 'IoU=0.5', 'IoU=0.75', 'small', 'medium', 'large']
    for k in range(6):
        ret['mAP(bbox)/' + fields[k]] = cocoEval.stats[k]

    if cfg.MODE_MASK:
        cocoEval = COCOeval(coco, cocoDt, 'segm')
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
        for k in range(6):
            ret['mAP(segm)/' + fields[k]] = cocoEval.stats[k]
    return ret
