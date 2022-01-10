class ObjectDetectorTrainConfig:
    train_image_folder: str
    annotation_fp: str
    dataset_name: str
    output_dir: str
    resume: bool
    max_iter: int
    num_workers: int
    ims_per_batch: int
    base_lr: int
    batch_size_per_image: int
    skip: bool

    def __init__(self, train_image_folder, annotation_fp, output_dir, categories, dataset_name='clevr', resume=True, max_iter=200,
                 num_workers=8, ims_per_batch=2, base_lr=0.01, batch_size_per_image=50, skip=False,
                 prediction_threshold=0.5):
        self.train_image_folder = train_image_folder
        self.annotation_fp = annotation_fp
        self.dataset_name = dataset_name
        self.output_dir = output_dir
        self.resume = resume
        self.max_iter = max_iter
        self.num_workers = num_workers
        self.ims_per_batch = ims_per_batch
        self.base_lr = base_lr
        self.batch_size_per_image = batch_size_per_image
        self.skip = skip
        self.prediction_threshold = prediction_threshold
        self.categories = categories
