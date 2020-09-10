import tensorflow as tf

from tensorflow.keras.optimizers import RMSprop

'''
:returns model, model.summary
'''
def model():

    model = tf.keras.models.Sequential([tf.keras.layers.Conv2D(32, (3,3),activation='relu', input_shape=(140,140,3)),
                                        tf.keras.layers.BatchNormalization(),
                                        tf.keras.layers.MaxPooling2D(2,2),
                                        tf.keras.layers.Dropout(0.25),

                                        tf.keras.layers.Conv2D(64,(3,3), activation='relu'),
                                        tf.keras.layers.BatchNormalization(),
                                        tf.keras.layers.MaxPooling2D(2, 2),
                                        tf.keras.layers.Dropout(0.25),

                                        tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
                                        tf.keras.layers.MaxPooling2D(2,2),

                                        tf.keras.layers.Flatten(),
                                        tf.keras.layers.Dense(512, activation='relu'),
                                        tf.keras.layers.Dense(2, activation='softmax')])

    model.compile(optimizer=RMSprop(),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model, model.summary()


