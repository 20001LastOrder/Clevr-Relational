from typing import List


class RelNetConfiguration:
    def __init__(self, run_dir: str, ann_path: str, img_h5: str, num_rels: int, split_id: int,
                 dev: bool = False, max_epochs: int = 20, precision: int = 32, batch_size: int = 64,
                 num_workers: int = 4, learning_rate: float = 0.002, resume_from_checkpoint=None,
                 label_names: List = None, model_path: str = '', scenes_path: str = '', use_proba: bool = True,
                 output_path: str = '', test_ann_path: str = '', test_img_h5: str = ''):
        self.run_dir = run_dir
        self.dev = dev
        self.max_epochs = max_epochs
        self.precision = precision
        self.ann_path = ann_path
        self.img_h5 = img_h5
        self.num_rels = num_rels
        self.split_id = split_id
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.learning_rate = learning_rate
        self.resume_from_checkpoint = resume_from_checkpoint

        # test configs
        self.label_names = label_names
        self.model_path = model_path
        self.scenes_path = scenes_path
        self.use_proba = use_proba
        self.output_path = output_path
        self.test_ann_path = test_ann_path
        self.test_img_h5 = test_img_h5
