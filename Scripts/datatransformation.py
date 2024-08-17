import tensorflow as tf

class DataTransformationPipeline:
    def __init__(self, image_size=(224, 224), feature_extractor=None):
        self.image_size = image_size
        self.feature_extractor = feature_extractor

    def preprocess_images(self, image, label):
        image = tf.cast(image, tf.float32) / 255.0  # Scale images to [0, 1]
        image = tf.image.resize(image, self.image_size)
        image = (image - self.feature_extractor.image_mean) / self.feature_extractor.image_std
        return image, label

    def transform(self, dataset):
        return dataset.map(self.preprocess_images)

# Example usage:
# transformer = DataTransformationPipeline(image_size=(224, 224), feature_extractor=feature_extractor)
# train_ds = transformer.transform(train_ds)
# val_ds = transformer.transform(val_ds)

