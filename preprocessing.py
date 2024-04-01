import wget
import zipfile
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator

class RockPaperScissorsDataset:
    def __init__(self, url, file_name, extract_to):
        self.url = url
        self.file_name = file_name
        self.extract_to = extract_to

    def download_file_if_not_exists(self):
        if not os.path.exists(self.file_name):
            wget.download(self.url)
            print("File downloaded successfully!")
        else:
            print("File has already been downloaded!")

    def extract_zipfile(self):
        with zipfile.ZipFile(self.file_name, 'r') as zip_ref:
            zip_ref.extractall(self.extract_to)
        print('Data has been extracted!.')

class ImagePreprocessor:
    def __init__(self, img_size=(150, 150)):
        self.img_size = img_size

    def preprocess_images(self, folder_path):
        images = []
        labels = []
        for label, category in enumerate(['rock', 'paper', 'scissors']):
            category_path = os.path.join(folder_path, category)
            for img_name in os.listdir(category_path):
                img_path = os.path.join(category_path, img_name)
                img = cv2.imread(img_path)
                img = cv2.resize(img, self.img_size)
                img = img / 255.0
                images.append(img)
                labels.append(label)
        return np.array(images), np.array(labels)

class DataGenerator:
    def __init__(self, train_images, train_labels, test_images, test_labels, batch_size=32):
        self.train_images = train_images
        self.train_labels = train_labels
        self.test_images = test_images
        self.test_labels = test_labels
        self.batch_size = batch_size

    def generate_train_data(self):
        datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            vertical_flip=True,
            fill_mode='nearest'
        )
        return datagen.flow(self.train_images, self.train_labels, batch_size=self.batch_size)

    def generate_test_data(self):
        datagen = ImageDataGenerator()
        return datagen.flow(self.test_images, self.test_labels, batch_size=self.batch_size)
    
    @staticmethod
    def visualize_augmented_images(datagen, sample_images, augmentation_names, save_folder='Result/AugmentationData'):
        plt.figure(figsize=(12, 10))
        for i, sample_image in enumerate(sample_images):
            sample_image = np.expand_dims(sample_image, axis=0)
            augmented_images = [sample_image]
            for j in range(len(augmentation_names) - 1):
                x_batch = next(datagen)[0]
                augmented_images.append(x_batch)

            for j in range(len(augmented_images)):
                plt.subplot(len(sample_images), len(augmentation_names), i * len(augmentation_names) + j + 1)
                plt.imshow(augmented_images[j][0])
                if i == 0:
                    plt.title(augmentation_names[j])
                plt.axis('off')
        plt.tight_layout()
        
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        save_path = os.path.join(save_folder, 'data-augmentation-visualizations.png')
        plt.savefig(save_path)
        plt.close()