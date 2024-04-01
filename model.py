import tensorflow as tf
from keras.callbacks import ModelCheckpoint

def build_sequential_model(input_img_shape=(150, 150, 3), num_classes=3):
    sequential_model = tf.keras.models.Sequential()

    sequential_model.add(tf.keras.layers.Conv2D(
        filters=64,
        kernel_size=(3, 3),
        activation='relu',
        input_shape=input_img_shape
    ))
    sequential_model.add(tf.keras.layers.MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2)
    ))

    sequential_model.add(tf.keras.layers.Conv2D(
        filters=64,
        kernel_size=(3, 3),
        activation='relu'
    ))
    sequential_model.add(tf.keras.layers.MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2)
    ))

    sequential_model.add(tf.keras.layers.Conv2D(
        filters=128,
        kernel_size=(3, 3),
        activation='relu'
    ))

    sequential_model.add(tf.keras.layers.MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2)
    ))

    sequential_model.add(tf.keras.layers.Conv2D(
        filters=128,
        kernel_size=(3, 3),
        activation='relu'
    ))

    sequential_model.add(tf.keras.layers.MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2)
    ))

    sequential_model.add(tf.keras.layers.Flatten())
    sequential_model.add(tf.keras.layers.Dropout(0.5))

    sequential_model.add(tf.keras.layers.Dense(
        units=512,
        activation='relu'
    ))

    sequential_model.add(tf.keras.layers.Dense(
        units=num_classes,
        activation='softmax'
    ))

    adam_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    sequential_model.compile(optimizer=adam_optimizer,
                             loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                             metrics=['accuracy'])
    
    return sequential_model

class CustomerCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        if logs.get('accuracy') > 0.96:
            self.model.stop_training = True
            print('\nThe accuracy threshold has been achieved, more than 96%!')
            
def train_model(model, train_generator, test_generator, train_steps_per_epoch, test_steps):
    checkpoint_filepath = 'best_model.keras'

    checkpoint = ModelCheckpoint(filepath=checkpoint_filepath,
                                 save_weights_only=False,
                                 monitor='val_accuracy',
                                 mode='max',
                                 save_best_only=True)

    history = model.fit(x=train_generator,
                        validation_data=test_generator,
                        epochs=20,
                        steps_per_epoch=train_steps_per_epoch,
                        validation_steps=test_steps,
                        verbose=1,
                        callbacks=[checkpoint, CustomerCallback()])
    return history

