from utils.simpleclass import SimpleClass
from pathlib import Path
from utils.metric import Metric
from utils.AP import AP
from utils.iou import IOU
import torch
from icecream import ic
import numpy as np
from ultralytics.utils import plt_settings
import matplotlib.pyplot as plt


class DetMetrics(SimpleClass):  # detection metrics
    """
    This class is a utility class for computing detection metrics such as precision, recall, and mean average precision
    (mAP) of an object detection model.

    Args:
        save_dir (Path): A path to the directory where the output plots will be saved. Defaults to current directory.
        plot (bool): A flag that indicates whether to plot precision-recall curves for each class. Defaults to False.
        on_plot (func): An optional callback to pass plots path and data when they are rendered. Defaults to None.
        names (tuple of str): A tuple of strings that represents the names of the classes. Defaults to an empty tuple.

    Attributes:
        save_dir (Path): A path to the directory where the output plots will be saved.
        plot (bool): A flag that indicates whether to plot the precision-recall curves for each class.
        on_plot (func): An optional callback to pass plots path and data when they are rendered.
        names (tuple of str): A tuple of strings that represents the names of the classes.
        box (Metric): An instance of the Metric class for storing the results of the detection metrics.
        speed (dict): A dictionary for storing the execution time of different parts of the detection process.

    Methods:
        process(tp, conf, pred_cls, target_cls): Updates the metric results with the latest batch of predictions.
        keys: Returns a list of keys for accessing the computed detection metrics.
        mean_results: Returns a list of mean values for the computed detection metrics.
        class_result(i): Returns a list of values for the computed detection metrics for a specific class.
        maps: Returns a dictionary of mean average precision (mAP) values for different IoU thresholds.
        fitness: Computes the fitness score based on the computed detection metrics.
        ap_class_index: Returns a list of class indices sorted by their average precision (AP) values.
        results_dict: Returns a dictionary that maps detection metric keys to their computed values.
        curves: TODO
        curves_results: TODO
    """

    def __init__(
        self,
        save_dir=Path("."),
        plot=False,
        on_plot=None,
        names=(),
        conf=0.001,
        #iou_threshold=0.5,
    ) -> None: 
        """Initialize a DetMetrics instance with a save directory, plot flag, callback function, and class names."""
        self.save_dir = save_dir
        self.plot = plot
        self.on_plot = on_plot
        self.names = names
        self.nc = len(names)
        self.box = Metric()
        self.speed = {
            "preprocess": 0.0,
            "inference": 0.0,
            "loss": 0.0,
            "postprocess": 0.0,
        }
        self.task = "detect"
        self.conf = conf
        # self.iou_threshold = iou_threshold
        self.iou_list = torch.linspace(start=0.5, end=0.95, steps=10)
        # self.collected_data = torch.tensor([])
        self.collected_data = [ torch.zeros(1,4) for _ in range(len(self.iou_list))]
        self.all_ap = torch.zeros([self.nc, len(self.iou_list)])
        # ic(self.collected_data)
        # ic(self.all_ap)

    def collect_results(self, gt_boxes=None, pred_boxes=None, gt_classes=None):
        """
        This function use to collect the confidence score and TP/FP for each class

        input:
            gt_boxes format (tensor)= x1,y1,x2,y2
            pred_boxes format (tensor) [x1,y1,x2,y2,conf,class]
            gt_classes (class of ground truth) tensor [# boxes, classs  ]
        output:
           tensor list of boxes : [conf, TP/FP, pred_class, target_class]
        """
        for iou_idx, _iou in enumerate(self.iou_list):
            if gt_classes.size(0) is None:  # check if lable is empty
                detections = pred_boxes[pred_boxes[:, 4] >= self.conf]
                for d in detections:
                    # ic(self.collected_data[iou_idx])
                    self.collected_data[iou_idx] = torch.cat(
                        (self.collected_data[iou_idx], torch.tensor([[d[4], False, d[5], self.nc]])),
                        dim=0,
                    )  # FP

            if pred_boxes is None:
                return  # FN

            detections = pred_boxes[pred_boxes[:, 4] >= self.conf]
            detection_classes = detections[:, 5].int()

            gt_classes = gt_classes.int()
            iou = IOU.box_iou(gt_boxes, detections[:, :4])

            x = torch.where(iou > _iou)

            if x[0].shape[0]:  #
                matches = (
                    torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1)
                    .cpu()
                    .numpy()
                )  # [row index, column index, iou]
                if x[0].shape[0] > 1:
                    matches = matches[
                        matches[:, 2].argsort()[::-1]
                    ]  # sort array from maximum iou to minimum iou
                    # np.unique(matches[:, 1], return_index=True)[1] = detections bounding boxes ต้องหา unique เนื่องจาก boudning box เดียว match ได้กับ grouth-truth เดียว
                    matches = matches[
                        np.unique(matches[:, 1], return_index=True)[1]
                    ]  # bounding box ที่ index อะไรบ้างที่ matches กับ grouth-truth
                    matches = matches[matches[:, 2].argsort()[::-1]]
                    matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            else:  # add to false positive of that class
                matches = np.zeros((0, 3))

            n = (
                matches.shape[0] > 0
            )  # true if found matches between ground truth and detections
            m0, m1, _ = matches.transpose().astype(int)
            # m0 = grouth truth index
            # m1 = detection index

            for i, gc in enumerate(gt_classes):
                j = m0 == i
                if n and sum(j) == 1:
                    # self.matrix[detection_classes[m1[j]], gc] += 1  # correct -> TP
                    # ic(self.collected_data[iou_idx], self.collected_data[iou_idx].shape )
                    # ic(torch.tensor([[detections[m1[j][0]][4], True, gc, gc]]),torch.tensor([[detections[m1[j][0]][4], True, gc, gc]]).shape)
                    self.collected_data[iou_idx] = torch.cat(
                        (
                            self.collected_data[iou_idx],
                            torch.tensor([[detections[m1[j][0]][4], True, gc, gc]]),
                        ),
                        dim=0,
                    )
                  
                else:
                    # self.matrix[self.nc, gc] += 1  # true background -> ไม่เจอ boudngin box ที่ match กับ ground-truth เลย -> FN
                    pass

            if n:
                for i, dc in enumerate(detection_classes):
                    if not any(m1 == i):  # box ที่เหลือ ที่มี iou < threshold
                        # self.matrix[dc, self.nc] += 1  # predicted background -> FP
                        self.collected_data[iou_idx] = torch.cat(
                            (
                                self.collected_data[iou_idx],
                                torch.tensor([[detections[i][4], False, dc, self.nc]]),
                            ),
                            dim=0,
                        )
                        
        return

    # def process(self, tp, conf, pred_cls, target_cls):
    def process(
        self,
    ):
        tp = self.collected_data[:, 1]
        conf = self.collected_data[:, 0]
        pred_cls = self.collected_data[:, 2]
        target_cls = self.collected_data[:, 3]
        """Process predicted results for object detection and update metrics."""
    
        results = ap_per_class(
            tp=tp.numpy(),
            conf=conf.numpy(),
            pred_cls=pred_cls.numpy(),
            target_cls=target_cls.numpy(),
            plot=self.plot,
            save_dir=self.save_dir,
            names=self.names,
            on_plot=self.on_plot,
            iou_threshold=self.iou_threshold,
        )[2:]
        self.box.nc = len(self.names) #remove background class
        self.box.update(results)

    @property
    def keys(self):
        """Returns a list of keys for accessing specific metrics."""
        return [
            "metrics/precision(B)",
            "metrics/recall(B)",
            "metrics/mAP50(B)",
            "metrics/mAP50-95(B)",
        ]

    def mean_results(self):
        """Calculate mean of detected objects & return precision, recall, mAP50, and mAP50-95."""
        return self.box.mean_results()

    def class_result(self, i):
        """Return the result of evaluating the performance of an object detection model on a specific class."""
        return self.box.class_result(i)

    @property
    def maps(self):
        """Returns mean Average Precision (mAP) scores per class."""
        return self.box.maps

    @property
    def fitness(self):
        """Returns the fitness of box object."""
        return self.box.fitness()

    @property
    def ap_class_index(self):
        """Returns the average precision index per class."""
        return self.box.ap_class_index

    @property
    def results_dict(self):
        """Returns dictionary of computed performance metrics and statistics."""
        return dict(zip(self.keys + ["fitness"], self.mean_results() + [self.fitness]))

    @property
    def curves(self):
        """Returns a list of curves for accessing specific metrics curves."""
        return [
            "Precision-Recall(B)",
            "F1-Confidence(B)",
            "Precision-Confidence(B)",
            "Recall-Confidence(B)",
        ]

    @property
    def curves_results(self):
        """Returns dictionary of computed performance metrics and statistics."""
        return self.box.curves_results


def ap_per_class(
    tp,
    conf,
    pred_cls,
    target_cls,
    plot=False,
    on_plot=None,
    save_dir=Path(),
    names=(),
    eps=1e-16,
    prefix="",
    iou_threshold=0.5,
):
    """
    Computes the average precision per class for object detection evaluation.

    Args:
        tp (np.ndarray): Binary array indicating whether the detection is correct (True) or not (False).
        conf (np.ndarray): Array of confidence scores of the detections.
        pred_cls (np.ndarray): Array of predicted classes of the detections.
        target_cls (np.ndarray): Array of true classes of the detections.
        plot (bool, optional): Whether to plot PR curves or not. Defaults to False.
        on_plot (func, optional): A callback to pass plots path and data when they are rendered. Defaults to None.
        save_dir (Path, optional): Directory to save the PR curves. Defaults to an empty path.
        names (tuple, optional): Tuple of class names to plot PR curves. Defaults to an empty tuple.
        eps (float, optional): A small value to avoid division by zero. Defaults to 1e-16.
        prefix (str, optional): A prefix string for saving the plot files. Defaults to an empty string.

    Returns:
        (tuple): A tuple of six arrays and one array of unique classes, where:
            tp (np.ndarray): True positive counts at threshold given by max F1 metric for each class.Shape: (nc,).
            fp (np.ndarray): False positive counts at threshold given by max F1 metric for each class. Shape: (nc,).
            p (np.ndarray): Precision values at threshold given by max F1 metric for each class. Shape: (nc,).
            r (np.ndarray): Recall values at threshold given by max F1 metric for each class. Shape: (nc,).
            f1 (np.ndarray): F1-score values at threshold given by max F1 metric for each class. Shape: (nc,).
            ap (np.ndarray): Average precision for each class at different IoU thresholds. Shape: (nc, 10).
            unique_classes (np.ndarray): An array of unique classes that have data. Shape: (nc,).
            p_curve (np.ndarray): Precision curves for each class. Shape: (nc, 1000).
            r_curve (np.ndarray): Recall curves for each class. Shape: (nc, 1000).
            f1_curve (np.ndarray): F1-score curves for each class. Shape: (nc, 1000).
            x (np.ndarray): X-axis values for the curves. Shape: (1000,).
            prec_values: Precision values at mAP@0.5 for each class. Shape: (nc, 1000).
    """

    # Sort by objectness
    i = np.argsort(-conf) #sort index from max -> min confidence
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes, nt = np.unique(target_cls, return_counts=True)
    nc = unique_classes.shape[0]  # number of classes, number of detections

    # Create Precision-Recall curve and compute AP for each class
    x, prec_values = np.linspace(0, 1, 1000), []

    # Average precision, precision and recall curves
    ap, p_curve, r_curve = (
        # np.zeros((nc, tp.shape[1])),
        np.zeros((nc, 1)), #FIXME: delete this line
        np.zeros((nc, 1000)),
        np.zeros((nc, 1000)),
    )

    for ci, c in enumerate(unique_classes):
        i = pred_cls == c # boolean list ที่ predict class = c
        n_l = nt[ci]  # number of labels
        n_p = i.sum()  # number of predictions
        if n_p == 0 or n_l == 0:
            continue

        # Accumulate FPs and TPs
        fpc = (1 - tp[i]).cumsum(0) #cumulative sum
        tpc = tp[i].cumsum(0)
        

        # Recall
        recall = tpc / (n_l + eps)  # recall curve

        r_curve[ci] = np.interp(
            -x, -conf[i], recall, left=0
        )  # negative x, xp because xp decreases
        
        # Precision
        precision = tpc / (tpc + fpc)  # precision curve
        # p_curve[ci] = np.interp(-x, -conf[i], precision[:, 0], left=1)  # p at pr_score
        p_curve[ci] = np.interp(-x, -conf[i], precision, left=1)  # p at pr_score

        # AP from recall-precision curve
        ap[ci], mpre, mrec = compute_ap(recall.T, precision.T)

        if plot:
            prec_values.append(np.interp(x, mrec, mpre))  # precision at mAP@0.5

    prec_values = np.array(prec_values)  # (nc, 1000)


    # Compute F1 (harmonic mean of precision and recall)
    f1_curve = 2 * p_curve * r_curve / (p_curve + r_curve + eps)
    ap = np.array(list([ _ap  for _ap in ap if float(_ap[0]) > 0]))
    if plot:
        plot_pr_curve(
            x,
            prec_values,
            ap,
            save_dir / f"{prefix}PR_curve.png",
            names,
            on_plot=on_plot,
            iou_threshold=iou_threshold,
        )
        plot_mc_curve(
            x,
            f1_curve,
            save_dir / f"{prefix}F1_curve.png",
            names,
            ylabel="F1",
            on_plot=on_plot,
        )
        plot_mc_curve(
            x,
            p_curve,
            save_dir / f"{prefix}P_curve.png",
            names,
            ylabel="Precision",
            on_plot=on_plot,
        )
        plot_mc_curve(
            x,
            r_curve,
            save_dir / f"{prefix}R_curve.png",
            names,
            ylabel="Recall",
            on_plot=on_plot,
        )

    i = smooth(f1_curve.mean(0), 0.1).argmax()  # max F1 index
    p, r, f1 = (
        p_curve[:, i],
        r_curve[:, i],
        f1_curve[:, i],
    )  # max-F1 precision, recall, F1 values
    tp = (r * nt).round()  # true positives
    fp = (tp / (p + eps) - tp).round()  # false positives
    # ic((
    #     tp,
    #     fp,
    #     p,
    #     r,
    #     f1,
    #     ap,
    #     # unique_classes.astype(int),
    #     ic(np.array(list(range(len(names)))).astype(int)),
    #     p_curve,
    #     r_curve,
    #     f1_curve,
    #     x,
    #     prec_values,
    # ))

    # ic((
    #     tp,
    #     fp,
    #     p,
    #     r,
    #     f1,
    #     ap,
    #     # unique_classes.astype(int),
    #     ic(np.array(list(range(len(names)))).astype(int)),
    #     p_curve,
    #     r_curve,
    #     f1_curve,
    #     x,
    #     prec_values,
    # )[2:])

    
    return (
        tp,
        fp,
        p,
        r,
        f1,
        ap,
        # unique_classes.astype(int),
        np.array(list(range(len(names)))).astype(int),
        p_curve,
        r_curve,
        f1_curve,
        x,
        prec_values,
    )

def compute_ap(recall, precision):
        """
        Compute the average precision (AP) given the recall and precision curves.

        Args:
            recall (list): The recall curve.
            precision (list): The precision curve.

        Returns:
            (float): Average precision.
            (np.ndarray): Precision envelope curve.
            (np.ndarray): Modified recall curve with sentinel values added at the beginning and end.
        """

        # Append sentinel values to beginning and end
        mrec = np.concatenate(([0.0], recall, [1.0]))
        mpre = np.concatenate(([1.0], precision, [0.0]))

        # Compute the precision envelope
        mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

        # Integrate area under curve
        method = "interp"  # methods: 'continuous', 'interp'
        if method == "interp":
            x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
            ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate
        else:  # 'continuous'
            i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x-axis (recall) changes
            ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve

        return ap, mpre, mrec
    
    
@plt_settings()
def plot_pr_curve(px, py, ap, save_dir=Path("pr_curve.png"), names=(), on_plot=None, iou_threshold=0.5):
    # ic(names)
    """Plots a precision-recall curve."""
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)
    py = np.stack(py, axis=1)

    if 0 < len(names) < 21:  # display per-class legend if < 21 classes
        for i, y in enumerate(py.T):
            ax.plot(px, y, linewidth=1, label=f"{names[i]} {ap[i, 0]:.3f}")  # plot(recall, precision)
    else:
        ax.plot(px, py, linewidth=1, color="grey")  # plot(recall, precision)
    # ic(f"plot {ap[:,0]}")
    ax.plot(px, py.mean(1), linewidth=3, color="blue", label=f"all classes %.3f mAP@{iou_threshold}" % ap[:, 0].mean())
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    ax.set_title("Precision-Recall Curve")
    fig.savefig(save_dir, dpi=250)
    plt.close(fig)
    if on_plot:
        on_plot(save_dir)


def smooth(y, f=0.05):
    """Box filter of fraction f."""
    nf = round(len(y) * f * 2) // 2 + 1  # number of filter elements (must be odd)
    p = np.ones(nf // 2)  # ones padding
    yp = np.concatenate((p * y[0], y, p * y[-1]), 0)  # y padded
    return np.convolve(yp, np.ones(nf) / nf, mode="valid")  # y-smoothed

@plt_settings()
def plot_mc_curve(px, py, save_dir=Path("mc_curve.png"), names=(), xlabel="Confidence", ylabel="Metric", on_plot=None):
    """Plots a metric-confidence curve."""
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)

    if 0 < len(names) < 21:  # display per-class legend if < 21 classes
        for i, names in enumerate(names):
            # ic(i, y, names[i])
            ax.plot(px, py[i], linewidth=1, label=f"{names}")  # plot(confidence, metric)
        
        # for i, y in enumerate(py):
        #     # ic(i, y, names[i])
        #     ax.plot(px, y, linewidth=1, label=f"{names[i]}")  # plot(confidence, metric)
    else:
        ax.plot(px, py.T, linewidth=1, color="grey")  # plot(confidence, metric)

    y = smooth(py.mean(0), 0.05)
    ax.plot(px, y, linewidth=3, color="blue", label=f"all classes {y.max():.2f} at {px[y.argmax()]:.3f}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    ax.set_title(f"{ylabel}-Confidence Curve")
    fig.savefig(save_dir, dpi=250)
    plt.close(fig)
    if on_plot:
        on_plot(save_dir)