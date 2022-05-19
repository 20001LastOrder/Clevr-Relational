from typing import List


class RelNetConfiguration:
    def __init__(self, run_dir: str, ann_path: str, img_h5: str, num_rels: int, train_size: int,
                 dev: bool = False, max_epochs: int = 20, precision: int = 32, batch_size: int = 64,
                 num_workers: int = 4, learning_rate: float = 0.002, resume_from_checkpoint=None,
                 label_names: List = None, model_path: str = '', scenes_path: str = '', use_proba: bool = True,
                 output_path: str = '', test_ann_path: str = '', test_img_h5: str = '', dropout_p: float = 0,
                 model_type: str = 'normal', noise_ratio: float = 0, include_constraint_loss: bool = False,
                 used_rels: List = None, use_sigmoid: bool = True, val_size=None, filename=None, shuffle_train=False,
                 seed=None):
        self.run_dir = run_dir
        self.dev = dev
        self.max_epochs = max_epochs
        self.precision = precision
        self.ann_path = ann_path
        self.img_h5 = img_h5
        self.num_rels = num_rels
        self.train_size = train_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.learning_rate = learning_rate
        self.resume_from_checkpoint = resume_from_checkpoint
        self.dropout_p = dropout_p
        self.shuffle_train = shuffle_train
        self.val_size = val_size
        self.seed = seed

        self.model_type = model_type
        self.noise_ratio = noise_ratio
        self.include_constraint_loss = include_constraint_loss
        self.used_rels = used_rels
        self.use_sigmoid = use_sigmoid
        self.filename = filename
        
        # test configs
        self.label_names = label_names
        self.model_path = model_path
        self.scenes_path = scenes_path
        self.use_proba = use_proba
        self.output_path = output_path
        self.test_ann_path = test_ann_path
        self.test_img_h5 = test_img_h5
