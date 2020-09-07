import Classifier.classifiermodel as clf
import Classifier.Dataloader
import tensorflow as tf
import os

base_dir = '../Datasets/dogs-vs-cats/'

model, summary = clf.model()


# Added Learning rate schedular.

def schedular(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.2)

print(summary)

train, validation, steps_per_epoc, validation_steps = Classifier.Dataloader.load_data(train_dir=os.path.join(base_dir, 'train'),
                                                                                      classmode='categorical',
                                                                                      batch_size=10
                                                                                      )

lrSchedular = tf.keras.callbacks.LearningRateScheduler(schedular)
earlyStoping = tf.keras.callbacks.EarlyStopping(patience=10)




history = model.fit(train,
                    epochs=10,
                    validation_data=validation,
                    validation_steps=validation_steps,  # Match steps with your batch for avoiding data generator error
                    steps_per_epoch=steps_per_epoc,
                    callbacks=[lrSchedular, earlyStoping]
                    )
import matplotlib.pyplot as plt

# plot train and test loss.
plt.plot(history['loss'])
plt.plot(history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()