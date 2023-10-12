class TrainParameters:
    def __init__(
        self,
        model_type=None,
        data_yaml_path=None,
        epochs=None,
        imgsz=None,
        batch=None,
        cache=None,
        patience=None,
        device=None,
        workers=None,
        resume=False,
    ):
        self.model_type = model_type
        self.data_yaml_path = data_yaml_path
        self.epochs = epochs
        self.imgsz = imgsz
        self.batch = batch
        self.cache = cache
        self.patience = patience
        self.device = device
        self.workers = workers
        self.resume = resume
    
    def get_model_type(self):
        return self.model_type
    
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
    
    def get_resume(self):
        return self.resume
    
    def comet_parameters(self,):
        return

