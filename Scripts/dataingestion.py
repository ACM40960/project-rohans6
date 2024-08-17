import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory

class DataIngestion:
    def __init__(self, dataset_dir, validation_split=0.2, image_size=(224, 224), batch_size=32, seed=42):
        self.dataset_dir = dataset_dir
        self.validation_split = validation_split
        self.image_size = image_size
        self.batch_size = batch_size
        self.seed = seed

    def load_data(self):
        train_ds = image_dataset_from_directory(
            self.dataset_dir,
            labels="inferred",
            label_mode="int",
            batch_size=self.batch_size,
            image_size=self.image_size,
            shuffle=True,
            validation_split=self.validation_split,
            subset="training",
            seed=self.seed
        )

        val_ds = image_dataset_from_directory(
            self.dataset_dir,
            labels="inferred",
            label_mode="int",
            batch_size=self.batch_size,
            image_size=self.image_size,
            shuffle=True,
            validation_split=self.validation_split,
            subset="validation",
            seed=self.seed
        )

        return train_ds, val_ds

# Example usage:
# ingestion = DataIngestion('Dataset/train')
# train_ds, val_ds = ingestion.load_data()

