import tensorflow as tf
class DataValidation:
    def __init__(self, train_ds, val_ds):
        self.train_ds = train_ds
        self.val_ds = val_ds

    def validate_datasets(self):
        # Simple validation checks
        train_batch_count = tf.data.experimental.cardinality(self.train_ds).numpy()
        val_batch_count = tf.data.experimental.cardinality(self.val_ds).numpy()

        print(f"Training batches: {train_batch_count}")
        print(f"Validation batches: {val_batch_count}")

        if train_batch_count == 0:
            raise ValueError("Training dataset is empty or improperly loaded.")
        if val_batch_count == 0:
            raise ValueError("Validation dataset is empty or improperly loaded.")
        
        # Additional checks can be added here, such as label consistency, image integrity, etc.
        print("Datasets validated successfully.")

# Example usage:
# validation = DataValidation(train_ds, val_ds)
# validation.validate_datasets()
