class Image(object):
    def __init__(self, image_path: str, image_id: str, image_classification: str):
        self.image_path = image_path
        self.image_id = image_id
        self.image_classification = image_classification
