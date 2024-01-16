import numpy as np
import torch
from icecream import ic
from ultralytics.utils import LOGGER, TryExcept, plt_settings
import matplotlib.pyplot as plt
import warnings
from pathlib import Path
class ConfusionMatrix():
    def __init__(self,num_classes: int, iou_threshold,conf=0.001, labels=None):
        self.nc = num_classes
        self.iou_threshold = iou_threshold
        self.conf = conf
        self.labels = labels
        self.matrix = np.zeros((self.nc+1, self.nc+1)) #detection task add background class to the matrix

    
    def update(self, gt_boxes, detections, gt_classes):
        from utils.iou import IOU
        """
        update confusion matrix for object detection task

        gt_boxes = (# grouth-truth boxes, 4(x1,y1,x2,y1)) ex, torch.Size([2, 4])
        detections = (# detections boxes, 6(x1,y1,x2,y2,conf,cls)) ex, torch.Size([2, 6])
        gt_classes = class each grouth-truth boxes ex, torch.Size([2])
        """

        detection_conf_idx = 4
        detection_cls_idx = 5
        if gt_classes.size(0) == 0: # check if label is empty
            if detections is not None:
                detections = detections[detections[:, detection_conf_idx] >= self.conf]
                detection_classes = detections[:,detection_cls_idx].int()
                for dc in detection_classes:
                    self.matrix[dc,self.nc] += 1 # false positives
            return
        
        if detections is None:
            gt_classes = gt_classes.int()
            for gc in gt_classes:
                self.matrix[self.nc, gc] += 1 # background FN
            return

        detections = detections[detections[:, detection_conf_idx] >= self.conf]
        detection_classes = detections[:, detection_cls_idx].int()

        gt_classes = gt_classes.int()        
        iou = IOU.box_iou(gt_boxes, detections[:,:4]) # row = ground truth, column = detection
        
        
        x = torch.where(iou > self.iou_threshold)
        """
            shape of x = [2, #number of true condition]
            x[0] = row index
            x[1] = column index
        """
        if x[0].shape[0]: # the value is the number of bb that iou > threshold
            # torch.stack(x, 1) = (row index,column index) = position of bounding boxes
            # iou[x[0], x[1]][:,None] = torch.reshape(iou[x[0], x[1]],(-1,1)) = make row tensor to column tensor
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy() # [row index, column index, iou] 
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]] # sort array from maximum iou to minimum iou
                # np.unique(matches[:, 1], return_index=True)[1] = detections bounding boxes ต้องหา unique เนื่องจาก boudning box เดียว match ได้กับ grouth-truth เดียว
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]] # bounding box ที่ index อะไรบ้างที่ matches กับ grouth-truth
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        else:
            matches = np.zeros((0, 3))

        n = matches.shape[0] > 0 # true if found matches between ground truth and detections
        m0, m1, _ = matches.transpose().astype(int)
        # m0 = grouth truth index
        # m1 = detection index

        for i, gc in enumerate(gt_classes):
            j = m0 == i
            if n and sum(j) == 1:
                self.matrix[detection_classes[m1[j]], gc] += 1  # correct -> TP
            else:
                self.matrix[self.nc, gc] += 1  # true background -> ไม่เจอ boudngin box ที่ match กับ ground-truth เลย -> FN

        if n:
            for i, dc in enumerate(detection_classes):
                if not any(m1 == i): # box ที่เหลือ ที่มี iou < threshold
                    self.matrix[dc, self.nc] += 1  # predicted background -> FP
    
    def matrix(self):
        #TODO: row = pred, col= ground-truth
        return self.matrix
    
    @TryExcept("WARNING ⚠️ ConfusionMatrix plot failure")
    @plt_settings()
    def plot(self, normalize=True, save_dir="", names=(), on_plot=None):
        """
        Plot the confusion matrix using seaborn and save it to a file.

        Args:
            normalize (bool): Whether to normalize the confusion matrix.
            save_dir (str): Directory where the plot will be saved.
            names (tuple): Names of classes, used as labels on the plot.
            on_plot (func): An optional callback to pass plots path and data when they are rendered.
        """
        import seaborn as sn
        
        names = self.labels if self.labels is not None else ()
        
        array = self.matrix / ((self.matrix.sum(0).reshape(1, -1) + 1e-9) if normalize else 1)  # normalize columns
        array[array < 0.005] = np.nan  # don't annotate (would appear as 0.00)

        fig, ax = plt.subplots(1, 1, figsize=(12, 9), tight_layout=True)
        nc, nn = self.nc, len(names)  # number of classes, names
        sn.set(font_scale=1.0 if nc < 50 else 0.8)  # for label size
        labels = (0 < nn < 99) and (nn == nc)  # apply names to ticklabels
        ticklabels = (list(names) + ["background"]) if labels else "auto"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # suppress empty matrix RuntimeWarning: All-NaN slice encountered
            sn.heatmap(
                array,
                ax=ax,
                annot=nc < 30,
                annot_kws={"size": 8},
                cmap="Blues",
                fmt=".2f" if normalize else ".0f",
                square=True,
                vmin=0.0,
                xticklabels=ticklabels,
                yticklabels=ticklabels,
            ).set_facecolor((1, 1, 1))
        title = "Confusion Matrix" + ( " Normalized" * normalize) + f' IOU threshold {self.iou_threshold:.2f}'
        ax.set_xlabel("True")
        ax.set_ylabel("Predicted")
        ax.set_title(title)
        plot_fname = Path(save_dir) / f'{title.lower().replace(" ", "_")}.png'
        fig.savefig(plot_fname, dpi=250)
        plt.close(fig)
        if on_plot:
            on_plot(plot_fname)

    
    def print(self):
        """Print the confusion matrix to the console."""
        for i in range(self.nc + 1):
            LOGGER.info(" ".join(map(str, self.matrix[i])))
