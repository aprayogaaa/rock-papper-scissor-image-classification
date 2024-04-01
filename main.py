import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import classification_report

from preprocessing import RockPaperScissorsDataset, ImagePreprocessor, DataGenerator
from model import build_sequential_model, train_model
from evaluation import Evaluation

def main():
    url = "https://github.com/dicodingacademy/assets/releases/download/release/rockpaperscissors.zip"
    file_name = "rockpaperscissors.zip"
    extract_to = 'rockpaperscissors'

    # Download and extract dataset
    dataset = RockPaperScissorsDataset(url, file_name, extract_to)
    dataset.download_file_if_not_exists()
    if not os.path.exists(extract_to):
        os.makedirs(extract_to)
        dataset.extract_zipfile()

    # Preprocess images
    folder_path = "rockpaperscissors/rockpaperscissors/rps-cv-images"
    preprocessor = ImagePreprocessor()
    images, labels = preprocessor.preprocess_images(folder_path)
    images, labels = shuffle(images, labels, random_state=42)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.4, random_state=42)

    # Create data generators
    data_generator = DataGenerator(X_train, y_train, X_test, y_test)
    train_generator = data_generator.generate_train_data()
    test_generator = data_generator.generate_test_data()

    # Visualize augmented images
    sample_images = [X_train[np.random.choice(np.where(y_train == label)[0])] for label in np.unique(y_train)[:3]]
    DataGenerator.visualize_augmented_images(train_generator, sample_images, ['Original', 'Rotation', 'Width Shift', 'Height Shift', 'Shear', 'Zoom', 'Horizontal Flip', 'Vertical Flip'])

    # Print dataset details
    print('\nThis data divided into train and test data. Here are the details:')
    print('Number of training samples:', len(X_train))
    print('Number of test samples:', len(X_test))
    print('')

    # Build and train model
    model = build_sequential_model()
    model.summary()

    batch_size = 32
    train_steps_per_epoch = len(X_train) // batch_size
    test_steps = len(X_test) // batch_size

    print('\nFitting train data into model...')
    history = train_model(model, train_generator, test_generator, train_steps_per_epoch, test_steps)
    print('\nBest model has been created!, namely best_model.keras')
     
    # Evaluate model
    print('\nCreating loss and accuracy train data...')
    all_accuracy, all_loss, train_accuracy, test_accuracy, roc_value_train, roc_value_test = Evaluation.evaluate_model(model, y_train, y_test, train_generator, test_generator)

    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    classification_report_str = classification_report(y_test, y_pred_classes)

    # Save evaluation results
    Evaluation.save_evaluation_results(all_accuracy, all_loss, train_accuracy, test_accuracy, roc_value_train, roc_value_test, classification_report_str, y_test, y_pred_classes)
    print('\nModel evaluation has been successfully created!')
    
    # Open model evaluation
    result_file_path = os.path.join('Result', 'MetricsEvaluation', 'model-evaluation.txt')
    print("\nModel evaluation results:")
    with open(result_file_path, 'r') as f:
        print(f.read())
        
    print('\nThe whole process has been completed!')

if __name__ == "__main__":
    main()
