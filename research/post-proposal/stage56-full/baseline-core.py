import numpy as np
import logging
logger=logging.getLogger(__name__)

def voc_ap(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap*100

def get_gt_of_cls(gt_boxes, cls):
    cls_gt_boxes = []
    for i in range(gt_boxes.shape[0]):
        if len(gt_boxes.shape) > 1 and int(gt_boxes[i, -1]) == cls:
            cls_gt_boxes.append(gt_boxes[i, :-1])
    return np.asarray(cls_gt_boxes)

def compute_iou(box, cls_gt_boxes):

    ious = np.zeros(cls_gt_boxes.shape[0])

    for m in range(cls_gt_boxes.shape[0]):
        gtbox = cls_gt_boxes[m]

        xmin = max(gtbox[0], box[0])
        ymin = max(gtbox[1], box[1])
        xmax = min(gtbox[2], box[2])
        ymax = min(gtbox[3], box[3])
        iw = np.maximum(xmax - xmin, 0.)
        ih = np.maximum(ymax - ymin, 0.)
        if iw > 0 and ih > 0:
            intsc = iw*ih
        else:
            intsc = 0.0
        union = (gtbox[2] - gtbox[0]) * (gtbox[3] - gtbox[1]) + \
            (box[2] - box[0]) * (box[3] - box[1]) - intsc
        ious[m] = intsc/union

    return ious

def evaluate_detections(gt_boxes, det_boxes, classes=[], iou_thresh=0.5):

    ap_strs = []
    num_frames = len(gt_boxes)
    logger.info('Evaluating for '+ str(num_frames) + ' frames')
    ap_all = np.zeros(len(classes), dtype=np.float32)
    # loop over each class 'cls'
    for cls_ind, class_name in enumerate(classes):
        scores = np.zeros(num_frames * 2000)
        istp = np.zeros(num_frames * 2000)
        det_count = 0
        num_postives = 0.0
        for nf in range(num_frames):  # loop over each frame 'nf'
                # if len(gt_boxes[nf])>0 and len(det_boxes[cls_ind][nf]):
            # get frame detections for class cls in nf
            frame_det_boxes = np.copy(det_boxes[cls_ind][nf])
            # get gt boxes for class cls in nf frame
            cls_gt_boxes = get_gt_of_cls(np.copy(gt_boxes[nf]), cls_ind)
            num_postives += cls_gt_boxes.shape[0]
            # check if there are dection for class cls in nf frame
            if frame_det_boxes.shape[0] > 0:
                # sort in descending order
                sorted_ids = np.argsort(-frame_det_boxes[:, -1])
                for k in sorted_ids:  # start from best scoring detection of cls to end
                    box = frame_det_boxes[k, :-1]  # detection bounfing box
                    score = frame_det_boxes[k, -1]  # detection score
                    ispositive = False  # set ispostive to false every time
                    # we can only find a postive detection
                    if cls_gt_boxes.shape[0] > 0:
                        # if there is atleast one gt bounding for class cls is there in frame nf
                        # compute IOU between remaining gt boxes
                        iou = compute_iou(box, cls_gt_boxes)
                        # and detection boxes
                        # get the max IOU window gt index
                        maxid = np.argmax(iou)
                        # check is max IOU is greater than detection threshold
                        if iou[maxid] >= iou_thresh:
                            ispositive = True  # if yes then this is ture positive detection
                            # remove assigned gt box
                            cls_gt_boxes = np.delete(cls_gt_boxes, maxid, 0)
                    # fill score array with score of current detection
                    scores[det_count] = score
                    if ispositive:
                        # set current detection index (det_count)
                        istp[det_count] = 1
                        #  to 1 if it is true postive example
                    det_count += 1
        
        if num_postives < 1:
            num_postives = 1

        scores = scores[:det_count]
        istp = istp[:det_count]
        argsort_scores = np.argsort(-scores)  # sort in descending order
        istp = istp[argsort_scores]  # reorder istp's on score sorting
        fp = np.cumsum(istp == 0)  # get false positives
        tp = np.cumsum(istp == 1)  # get  true positives
        fp = fp.astype(np.float64)
        tp = tp.astype(np.float64)
        recall = tp / float(num_postives)  # compute recall
        # compute precision
        precision = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        # compute average precision using voc2007 metric
        cls_ap = voc_ap(recall, precision)
        ap_all[cls_ind] = cls_ap
        ap_str = class_name + ' : ' + \
            str(num_postives) + ' : ' + str(det_count) + ' : ' + str(cls_ap)
        ap_strs.append(ap_str)

    mAP = np.mean(ap_all)
    logger.info('Mean ap '+ str(mAP))
    return mAP, ap_all, ap_strs

def evaluate(gts, dets, all_classes, iou_thresh=0.5):
    # np.mean(ap_all), ap_all, ap_strs
    aps, aps_all, ap_strs = [], [], []
    for nlt in range(len(gts)):
        a, b, c = evaluate_detections(
            gts[nlt], dets[nlt], all_classes[nlt], iou_thresh)
        aps.append(a)
        aps_all.append(b)
        ap_strs.append(c)
    return aps, aps_all, ap_strs

def get_individual_labels(gt_boxes, tgt_labels):
    # print(gt_boxes.shape, tgt_labels.shape)
    new_gts = np.zeros((gt_boxes.shape[0]*20, 5))
    ccc = 0
    for n in range(tgt_labels.shape[0]):
        for t in range(tgt_labels.shape[1]):
            if tgt_labels[n,t]>0:
                new_gts[ccc, :4] = gt_boxes[n,:]
                new_gts[ccc, 4] = t
                ccc += 1
    return new_gts[:ccc,:]
