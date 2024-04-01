import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf

class Prediction:
    def __init__(self, model_path='best_model.keras'):
        self.model = self.load_model(model_path)
        self.root = tk.Tk()
        self.root.title("Rock Paper Scissors Prediction")

        self.select_button = tk.Button(self.root, text="Select Image", command=self.predict_image)
        self.select_button.pack(pady=20)

        self.result_label = tk.Label(self.root, text="")
        self.result_label.pack()

        self.image_label = tk.Label(self.root)
        self.image_label.pack()

    def load_model(self, model_path):
        model = tf.keras.models.load_model(model_path)
        return model

    def preprocess_image(self, image_path):
        img = Image.open(image_path)
        img = img.resize((150, 150))  
        img = np.array(img) / 255.0  
        img = np.expand_dims(img, axis=0) 
        return img

    def predict_image(self):
        file_path = filedialog.askopenfilename()

        preprocessed_img = self.preprocess_image(file_path)

        prediction = self.model.predict(preprocessed_img)
        class_names = ['rock', 'paper', 'scissors']
        predicted_class_index = np.argmax(prediction)
        predicted_class = class_names[predicted_class_index]
        probability = prediction[0][predicted_class_index]

        self.result_label.config(text=f"Prediction: {predicted_class}, Probability: {probability:.2f}")

        img = Image.open(file_path)
        img = img.resize((300, 300))
        img = ImageTk.PhotoImage(img)
        self.image_label.config(image=img)
        self.image_label.image = img

    def run(self):
        self.root.mainloop()

if __name__ == '__main__':
    prediction = Prediction()
    prediction.run()
