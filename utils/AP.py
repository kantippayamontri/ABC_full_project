import numpy as np
import matplotlib.pyplot as plt
from ultralytics.utils import plt_settings
from pathlib import Path

class AP:

    def __init__(self):
        return

    @plt_settings()
    def plot_pr_curve(self, px, py, ap, save_dir=Path("pr_curve.png"), names=(), on_plot=None):
        """Plots a precision-recall curve."""
        fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)
        py = np.stack(py, axis=1)

        if 0 < len(names) < 21:  # display per-class legend if < 21 classes
            for i, y in enumerate(py.T):
                ax.plot(px, y, linewidth=1, label=f"{names[i]} {ap[i, 0]:.3f}")  # plot(recall, precision)
        else:
            ax.plot(px, py, linewidth=1, color="grey")  # plot(recall, precision)

        ax.plot(px, py.mean(1), linewidth=3, color="blue", label="all classes %.3f mAP@0.5" % ap[:, 0].mean())
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

    @staticmethod
    def smooth(self,y, f=0.05):
        """Box filter of fraction f."""
        nf = round(len(y) * f * 2) // 2 + 1  # number of filter elements (must be odd)
        p = np.ones(nf // 2)  # ones padding
        yp = np.concatenate((p * y[0], y, p * y[-1]), 0)  # y padded
        return np.convolve(yp, np.ones(nf) / nf, mode="valid")  # y-smoothed

    @plt_settings()
    def plot_mc_curve(self,px, py, save_dir=Path("mc_curve.png"), names=(), xlabel="Confidence", ylabel="Metric", on_plot=None):
        """Plots a metric-confidence curve."""
        fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)

        if 0 < len(names) < 21:  # display per-class legend if < 21 classes
            for i, y in enumerate(py):
                ax.plot(px, y, linewidth=1, label=f"{names[i]}")  # plot(confidence, metric)
        else:
            ax.plot(px, py.T, linewidth=1, color="grey")  # plot(confidence, metric)

        y = self.smooth(py.mean(0), 0.05)
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


    @staticmethod
    def compute_ap(self,recall, precision):
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

    @staticmethod
    def ap_per_class(self,
        tp, conf, pred_cls, target_cls, plot=False, on_plot=None, save_dir=Path(), names=(), eps=1e-16, prefix=""
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
        i = np.argsort(-conf)
        tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

        # Find unique classes
        unique_classes, nt = np.unique(target_cls, return_counts=True)
        nc = unique_classes.shape[0]  # number of classes, number of detections

        # Create Precision-Recall curve and compute AP for each class
        x, prec_values = np.linspace(0, 1, 1000), []

        # Average precision, precision and recall curves
        ap, p_curve, r_curve = np.zeros((nc, tp.shape[1])), np.zeros((nc, 1000)), np.zeros((nc, 1000))
        for ci, c in enumerate(unique_classes):
            i = pred_cls == c
            n_l = nt[ci]  # number of labels
            n_p = i.sum()  # number of predictions
            if n_p == 0 or n_l == 0:
                continue

            # Accumulate FPs and TPs
            fpc = (1 - tp[i]).cumsum(0)
            tpc = tp[i].cumsum(0)

            # Recall
            recall = tpc / (n_l + eps)  # recall curve
            r_curve[ci] = np.interp(-x, -conf[i], recall[:, 0], left=0)  # negative x, xp because xp decreases

            # Precision
            precision = tpc / (tpc + fpc)  # precision curve
            p_curve[ci] = np.interp(-x, -conf[i], precision[:, 0], left=1)  # p at pr_score

            # AP from recall-precision curve
            for j in range(tp.shape[1]):
                ap[ci, j], mpre, mrec = self.compute_ap(recall[:, j], precision[:, j])
                if plot and j == 0:
                    prec_values.append(np.interp(x, mrec, mpre))  # precision at mAP@0.5

        prec_values = np.array(prec_values)  # (nc, 1000)

        # Compute F1 (harmonic mean of precision and recall)
        f1_curve = 2 * p_curve * r_curve / (p_curve + r_curve + eps)
        names = [v for k, v in names.items() if k in unique_classes]  # list: only classes that have data
        names = dict(enumerate(names))  # to dict
        if plot:
            self.plot_pr_curve(x, prec_values, ap, save_dir / f"{prefix}PR_curve.png", names, on_plot=on_plot)
            self.plot_mc_curve(x, f1_curve, save_dir / f"{prefix}F1_curve.png", names, ylabel="F1", on_plot=on_plot)
            self.plot_mc_curve(x, p_curve, save_dir / f"{prefix}P_curve.png", names, ylabel="Precision", on_plot=on_plot)
            self.plot_mc_curve(x, r_curve, save_dir / f"{prefix}R_curve.png", names, ylabel="Recall", on_plot=on_plot)

        i = self.smooth(f1_curve.mean(0), 0.1).argmax()  # max F1 index
        p, r, f1 = p_curve[:, i], r_curve[:, i], f1_curve[:, i]  # max-F1 precision, recall, F1 values
        tp = (r * nt).round()  # true positives
        fp = (tp / (p + eps) - tp).round()  # false positives
        return tp, fp, p, r, f1, ap, unique_classes.astype(int), p_curve, r_curve, f1_curve, x, prec_values