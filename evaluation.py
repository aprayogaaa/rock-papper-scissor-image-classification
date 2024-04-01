import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score

class Evaluation:
    @staticmethod
    def render_training_history(training_history, save_folder='Result/MetricsEvaluation'):
        """
        Renders the training history with loss and accuracy plots.
        """
        loss = training_history.history['loss']
        val_loss = training_history.history['val_loss']
        accuracy = training_history.history['accuracy']
        val_accuracy = training_history.history['val_accuracy']

        plt.figure(figsize=(14, 4))

        plt.subplot(1, 2, 1)
        plt.title('Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.plot(loss, label='Training set')
        plt.plot(val_loss, label='Test set', linestyle='--')
        plt.legend()
        plt.grid(linestyle='--', linewidth=1, alpha=0.5)

        plt.subplot(1, 2, 2)
        plt.title('Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.plot(accuracy, label='Training set')
        plt.plot(val_accuracy, label='Test set', linestyle='--')
        plt.legend()
        plt.grid(linestyle='--', linewidth=1, alpha=0.5)

        plt.show()
        
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        save_path = os.path.join(save_folder, 'training-history.png')
        plt.savefig(save_path)
        plt.close()

    @staticmethod
    def evaluate_model(model, y_train_true, y_test_true, train_generator, test_generator):
        train_loss, train_accuracy = model.evaluate(train_generator)
        test_loss, test_accuracy = model.evaluate(test_generator)
        all_loss = (train_loss * len(y_train_true) + test_loss * len(y_test_true)) / (len(y_train_true) + len(y_test_true))
        all_accuracy = (train_accuracy * len(y_train_true) + test_accuracy * len(y_test_true)) / (len(y_train_true) + len(y_test_true))
        
        y_train_pred = model.predict(train_generator)
        y_test_pred = model.predict(test_generator)

        roc_value_train = roc_auc_score(y_train_true, y_train_pred, multi_class='ovo')
        roc_value_test = roc_auc_score(y_test_true, y_test_pred, multi_class='ovo')

        return all_accuracy, all_loss, train_accuracy, test_accuracy, roc_value_train, roc_value_test

    @staticmethod
    def save_evaluation_results(all_accuracy, all_loss, train_accuracy, test_accuracy, roc_value_train, roc_value_test, classification_report_str, y_true, y_pred, save_folder='Result/MetricsEvaluation'):
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        with open(os.path.join(save_folder, 'model-evaluation.txt'), 'w') as f:
            f.write(f'All accuracy: {all_accuracy:.2%}\n')
            f.write(f'All loss: {all_loss:.2%}\n')
            f.write(f'Train accuracy: {train_accuracy:.2%}\n')
            f.write(f'Test accuracy: {test_accuracy:.2%}\n')
            f.write(f'ROC value train: {roc_value_train:.2%}\n')
            f.write(f'ROC value test: {roc_value_test:.2%}\n')
            f.write('\nClassification Report\n')
            f.write(classification_report_str)
            f.write('\nConfusion Matrix\n')
            f.write(str(confusion_matrix(y_true, y_pred)))