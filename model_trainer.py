'''
To-do list:
> read ID's from file
> read models layers from file
> assets from file
> save model training process and results to file
> save confusion matrix to file
> save model to file
> write scropt to read and compile these saved files information
'''


from pathlib import Path
import numpy as np
from sklearn.metrics import confusion_matrix
import itertools
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.metrics import *
import matplotlib.pyplot as plt


'''
Functions
'''

def plotImages(images_arr):
    fig, axes = plt.subplots(1, 10, figsize=(25,25))
    axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(cm, classes, normalize=False,
                          title='Confusion matrix', cmap=plt.cm.Blues):
    '''
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    '''
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def GPU_config(memory_growth):
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], memory_growth)
    print('Num GPUs available: ', len(physical_devices))


def gen_img_data(step, target_size, data_path, rescale = 1/255, zoom_range = 0,
                      batch_size = 100, class_mode = 'categorical'):
    datagen = ImageDataGenerator(rescale=rescale, zoom_range=zoom_range)
    batches = eval(f"datagen.flow_from_directory('{data_path}/{step}', target_size=target_size, batch_size=batch_size, class_mode=class_mode)")
    return batches


def build_model(target_size, layers):
    model = Sequential()
    model.add(Input(shape=(target_size[0], target_size[1], 3)))
    for layer in layers:
        eval(f'model.add({layer})')
    return model


def make_predictions(model, test_batches, model_id):
    predictions = model.predict(x=test_batches)
    cm = confusion_matrix(y_true=test_batches.classes, y_pred=np.argmax(predictions, axis=-1))
    plot_confusion_matrix(cm=cm, classes=list(test_batches.class_indices.keys()),
                          title=f'Confusion Matrix {model_id}')


'''
Main function
'''

def main():
    models_ids = ['00'] # Will be read from file
    models_layers = [["Conv2D(filters=16, kernel_size=11, strides=5, padding='same', activation='relu')", # Will be read from file
                      "BatchNormalization()",
                      "MaxPool2D(pool_size=4, strides=2, padding='same')",
                      "Flatten()",
                      "Dense(512, activation='relu')",
                      "Dropout(0.4)",
                      "Dense(128, activation='relu')",
                      "Dropout(0.4)",
                      "Dense(3, activation='softmax')"]]

    if len(models_ids) != len(models_layers):
        raise Exception('IDs and layers with different dimensions.')
    
    assets = ['VALE3'] # Will be read from file
    steps = ['train', 'test', 'valid']
    target_size = (300,300)
    epochs = 2

    GPU_config(memory_growth=True)

    for asset in assets:
        data_path = Path(f'./generated_data/{asset}')
        train_batches = gen_img_data(steps[0], target_size, data_path)
        valid_batches = gen_img_data(steps[1], target_size, data_path)
        test_batches  = gen_img_data(steps[2], target_size, data_path)
        for i in range(0, len(models_layers)):
            model = build_model(target_size, models_layers[i])
            model.compile(loss=categorical_crossentropy, optimizer='adam', metrics='accuracy')
            model.fit(x=train_batches, validation_data=valid_batches, epochs=epochs)
            predictions = make_predictions(model, test_batches, models_ids[i])


if __name__ == '__main__':
    main()
