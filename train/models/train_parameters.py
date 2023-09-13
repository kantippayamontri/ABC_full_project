class TrainParameters:
    def __init__(
        self,
        data_yaml_path=None,
        epochs=None,
        imgsz=None,
        batch=None,
        cache=None,
        patience=None,
        device=None,
        workers=None,
    ):
        self.data_yaml_path = data_yaml_path
        self.epochs = epochs
        self.imgsz = imgsz
        self.batch = batch
        self.cache = cache
        self.patience = patience
        self.device = device
        self.workers = workers
    
    def get_data_yaml_path(self):
        return self.data_yaml_path

    def get_epochs(self):
        return self.epochs

    def get_imgsz(self):
        return self.imgsz

    def get_batch(self):
        return self.batch

    def get_cache(self):
        return self.cache

    def get_patience(self):
        return self.patience

    def get_device(self):
        return self.device

    def get_workers(self):
        return self.workers
