from typing import List


class AttrNetConfiguration:
    def __init__(self, run_dir: str, dataset: str, load_checkpoint_path: str, ann_path: str, img_h5: str,
                 attr_names: List[str], output_dims: List[int], train_size: int = 3500, batch_size: int = 50,
                 num_workers: int = 8, learning_rate: int = 0.01, concat_img: bool = False, max_epochs: int = 20,
                 precision: int = 32, dev: bool = False, model_path: str = '', attr_map_path: str = '',
                 use_proba: bool = True, output_path: str = '', test_img_h5: str = '', test_ann_path: str = '',
                 desc: str = '', include_coords: bool = False, val_size=None, filename=None, resume_from_checkpoint=None,
                 check_val_every_n_epoch=10, seed=None):
        self.run_dir = run_dir
        self.dataset = dataset
        self.load_checkpoint_path = load_checkpoint_path
        self.ann_path = ann_path
        self.img_h5 = img_h5
        self.attr_names = attr_names
        self.output_dims = output_dims
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.include_coords = include_coords
        self.seed = seed

        # Training configuration
        self.train_size = train_size
        self.val_size = val_size
        self.learning_rate = learning_rate
        self.concat_img = concat_img
        self.max_epochs = max_epochs
        self.precision = precision
        self.dev = dev
        self.resume_from_checkpoint = resume_from_checkpoint
        self.check_val_every_n_epoch = check_val_every_n_epoch

        # Test configuration
        self.test_img_h5 = test_img_h5
        self.test_ann_path = test_ann_path
        self.model_path = model_path
        self.attr_map_path = attr_map_path
        self.use_proba = use_proba
        self.output_path = output_path
        self.desc = desc
        self.filename = filename
