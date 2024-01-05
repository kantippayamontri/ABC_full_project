class ConfusionMatrix():
    def __init__(self,num_classes: int, true_bb,pred_bb, iou_threshold):
        self.num_classes = num_classes
        self.true_bb = true_bb
        self.pred_bb = pred_bb
        self.iou_threshold = iou_threshold
    
    def calculate(self, ):
        return