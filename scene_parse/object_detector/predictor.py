import torch
import detectron2.data.transforms as T
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer


class ObjectPredictor:
    def __init__(self, cfg):
        self.cfg = cfg.clone()
        self.model = build_model(self.cfg)
        self.model.eval()
        
        if len(cfg.DATASETS.TEST):
            this.metadata = MetadataCatelog.get(cfg.DATASETS.TEST[0])
        
        DetectionCheckpointer(self.model).load(cfg.MODEL.WEIGHTS)
        self.aug = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MIN_SIZE_TEST
        )
        
        self.input_format = cfg.INPUT.FORMAT
        assert self.input_format in ['RGB', 'BGR'], self.input_format
    
    def get_bboxes(self, preprocessed_images, features):
        proposals, _ = self.model.proposal_generator(preprocessed_images, features)
        features_ = [features[f] for f in self.model.roi_heads.box_in_features]
        box_features = self.model.roi_heads.box_pooler(features_, [x.proposal_boxes for x in proposals])
        box_features = self.model.roi_heads.box_head(box_features)  # features of all 1k candidates
        
        return proposals, box_features
    
    def get_instances(self, preprocessed_images, box_features, proposals, features, inputs):
        predictions = self.model.roi_heads.box_predictor(box_features)
        pred_instances, pred_ids = self.model.roi_heads.box_predictor.inference(predictions, proposals)

        pred_instances = self.model.roi_heads.forward_with_given_boxes(features, pred_instances)
        pred_instances = self.model._postprocess(pred_instances, inputs, preprocessed_images.image_sizes)
        
        return pred_instances, pred_ids
        
    def __call__(self, original_image):
        with torch.no_grad():
            # the input image is expected to be BGR
            if self.input_format == 'RGB':
                # transform BGR to RGB
                original_image = original_image[:, :, ::-1]
                
            height, width = original_image.shape[:2]
            
            # transform image
            image = self.aug.get_transform(original_image).apply_image(original_image)
            image = torch.as_tensor(image.astype('float32').transpose(2, 0, 1))
            inputs = [{"image": image, "height": height, "width": width}]
            
            # get features of backbone model
            preprocessed_images = self.model.preprocess_image(inputs)
            features = self.model.backbone(preprocessed_images.tensor)
            
            proposals, box_features = self.get_bboxes(preprocessed_images, features)
            pred_instances, pred_ids = self.get_instances(preprocessed_images, box_features, proposals, features, inputs)

            return pred_instances[0], box_features[pred_ids]
        
def get_object_predictor(cfg, weight_path, score_threshold):
    cfg.MODEL.WEIGHTS = weight_path  # path to the model we just trained
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_threshold   # set a custom testing threshold
    return ObjectPredictor(cfg)