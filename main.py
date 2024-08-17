import os
from Scripts.get_data import DataDownloader
from Scripts.dataingestion import DataIngestion
from Scripts.datavalidation import DataValidation
from Scripts.datatransformation import DataTransformationPipeline
from Scripts.modeltrainer import ModelTrainer

def main():
    # Define constants
    dataset_url = 'https://drive.google.com/uc?export=download&id=1tQZFfpEcazxvrUuoqnbYfKiHY51X0ody'
    download_path = 'Dataset.zip'
    extract_to = 'Dataset'
    image_size = (224, 224)
    batch_size = 32
    num_classes = 4
    learning_rate = 5e-5
    epochs = 10
    savedmodels = os.path.join('savedmodels')

    # Step 0: Downloading Dataset
    print("Step 0: Downloading Dataset")
    downloader = DataDownloader(dataset_url, download_path, extract_to)
    downloader.download_and_extract()

    # Step 1: Data Ingestion
    print("Step 1: Data Ingestion")
    ingestion = DataIngestion(extract_to, image_size=image_size, batch_size=batch_size)
    train_ds, val_ds = ingestion.load_data()

    # Step 2: Data Validation
    print("Step 2: Data Validation")
    validation = DataValidation(train_ds, val_ds)
    validation.validate_datasets()

    # Step 3: Data Transformation
    print("Step 3: Data Transformation")
    from transformers import ViTFeatureExtractor
    feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
    transformer = DataTransformationPipeline(image_size=image_size, feature_extractor=feature_extractor)
    train_ds = transformer.transform(train_ds)
    val_ds = transformer.transform(val_ds)

    # Step 4: Model Training
    print("Step 4: Model Training")
    model_trainer = ModelTrainer(num_classes=num_classes, learning_rate=learning_rate)
    history = model_trainer.train(train_ds, val_ds, epochs=epochs)

    # Step 5: Evaluation Metrics Calculation
    print("Step 5: Evaluation")
    y_true, y_pred, y_pred_prob = model_trainer.evaluate(val_ds)
    
    # Calculate F1 Score, Precision, and Recall
    f1 = tf.keras.metrics.F1Score(num_classes=num_classes, average='weighted')
    precision = tf.keras.metrics.Precision()
    recall = tf.keras.metrics.Recall()

    f1.update_state(y_true, y_pred)
    precision.update_state(y_true, y_pred)
    recall.update_state(y_true, y_pred)

    print(f'F1 Score: {f1.result().numpy():.4f}')
    print(f'Precision: {precision.result().numpy():.4f}')
    print(f'Recall: {recall.result().numpy():.4f}')

    # Saving model
    os.makedirs(savedmodels, exist_ok=True)
    model = model_trainer.get_model()
    model.save(os.path.join(savedmodels, 'drowsiness.h5'))

if __name__ == "__main__":
    main()

