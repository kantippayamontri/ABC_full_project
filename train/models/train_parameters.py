class TrainParameters:
    def __init__(
        self,
        gauge_type=None,
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
        learning_rate=None,
        final_learning_rate=None,
    ):
        self.gauge_type = gauge_type
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
        self.learning_rate = learning_rate
        self.final_learning_rate = final_learning_rate
    
    def get_gauge_type(self):
        return self.gauge_type.value
    
    def get_model_type(self):
        return self.model_type.value
    
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
    
    def get_learning_rate(self):
        return self.learning_rate
    
    def get_final_learning_rate(self):
        return self.final_learning_rate
    
    def comet_parameters(self,):
        return {
            "gauge_type": self.get_gauge_type(),
            "model_type": self.get_model_type(),
            "epochs": self.get_epochs(),
            "imgsz": self.get_imgsz(),
            "batch": self.get_batch(),
            "cache": self.get_cache(),
            "patience": self.get_patience(),
            "device": self.get_device(),
            "workers": self.get_workers(),
            "resume": self.get_resume(),
            "learning_rate": self.get_learning_rate(),
            "final_learning_rate": self.get_final_learning_rate(),
        }

