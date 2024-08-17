import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling1D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from transformers import TFViTForImageClassification

class ModelTrainer:
    def __init__(self, num_classes=4, learning_rate=5e-5):
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.model = self._build_model()

    def _build_model(self):
        # Load the pre-trained ViT model
        base_model = TFViTForImageClassification.from_pretrained(
            'google/vit-base-patch16-224-in21k', num_labels=self.num_classes
        )
        base_model.trainable = False
        
        # Define the input
        inputs = tf.keras.Input(shape=(224, 224, 3))
        transposed_inputs = tf.transpose(inputs, perm=[0, 3, 1, 2])
        vit_outputs = base_model.vit(transposed_inputs)[0]

        # Add custom layers on top of ViT
        x = GlobalAveragePooling1D()(vit_outputs)
        x = Dropout(0.2)(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.2)(x)
        outputs = Dense(self.num_classes, activation='softmax')(x)

        # Build the model
        model = Model(inputs=inputs, outputs=outputs)

        # Compile the model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            metrics=['accuracy']
        )
        
        return model

    def train(self, train_ds, val_ds, epochs=10, callbacks=None):
        if callbacks is None:
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
                ModelCheckpoint('best_vit_model_with_custom_layers.h5', monitor='val_loss', save_best_only=True)
            ]

        # Train the model
        history = self.model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            callbacks=callbacks
        )

        return history

    def evaluate(self, test_ds):
        # Evaluate the model on the test dataset
        results = self.model.evaluate(test_ds)
        return results

    def get_model(self):
        return self.model

# Example usage:
# trainer = ModelTrainer(num_classes=4, learning_rate=5e-5)
# history = trainer.train(train_ds, val_ds, epochs=10)
# evaluation_results = trainer.evaluate(test_ds)
