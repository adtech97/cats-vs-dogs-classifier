from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import pandas as pd
import sklearn.model_selection as ms


'''
<p> This function categorises and augments the dataset<br>.

:returns training_data, testing_data, validation_steps and training_steps per epoch
'''
def load_data(train_dir: str, classmode: str, batch_size: int, debug: object = False):
    categories = []
    files = os.listdir(train_dir)
    for filename in files:
        _label = filename.split('.')[0]
        if _label == 'dog':
            categories.append(1)
        else:
            categories.append(0)

    data = pd.DataFrame({
        'files': files,
        'classes': categories
    })

    # Temporary fix for error convert column type as string
    data['classes'] = data['classes'].astype(str)
    # Check for the dataframe check last 5 elements
    if debug is True:
        print(data.tail(5))
    train, validate = ms.train_test_split(data, test_size=0.3, random_state=30)

    train = train.reset_index(drop=True)
    validate = validate.reset_index(drop=True)

    # Fix is for validation ant training steps calculations

    steps_per_epoc = train.shape[0] // batch_size
    validation_steps = validate.shape[0] // batch_size


    # Data is augmented and auto labeled to avoid over-fitting .
    train_datagen = ImageDataGenerator(rescale=1. / 255,
                                       horizontal_flip=True,
                                       rotation_range=60,
                                       width_shift_range=.2,
                                       height_shift_range=.2,
                                       zoom_range=0.45,
                                       channel_shift_range=80)

    training_data = train_datagen.flow_from_dataframe(train,
                                                      train_dir,
                                                      batch_size=batch_size,
                                                      x_col='files',
                                                      y_col='classes',
                                                      target_size=(140, 140),
                                                      class_mode=classmode)

    test_datagen = ImageDataGenerator(rescale=1. / 255)


    testing_data = test_datagen.flow_from_dataframe(validate,
                                                    train_dir,
                                                    x_col='files',
                                                    y_col='classes',
                                                    batch_size=batch_size,
                                                    target_size=(140, 140),
                                                    class_mode=classmode)

    return training_data, testing_data, steps_per_epoc, validation_steps
