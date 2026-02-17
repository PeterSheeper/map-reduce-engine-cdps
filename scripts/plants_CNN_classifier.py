import datetime
from pathlib import Path

from PIL import Image
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score
from tensorflow import numpy_function
from tensorflow.data import Dataset
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ConvNeXtTiny
from tensorflow.keras.applications.convnext import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import img_to_array

from tasks.plants_classifier import build_dataset

IMG_SIZE = (224, 224)
BATCH_SIZE = 32


def build_model(num_classes):
    base_model = ConvNeXtTiny(
        weights='imagenet',
        include_top=False,
        input_shape=(*IMG_SIZE, 3)
    )
    base_model.trainable = False

    return models.Sequential([
        layers.Input(shape=(*IMG_SIZE, 3)),
        layers.Lambda(preprocess_input),
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(num_classes, activation='softmax')
    ])


def train(model, train_dataset, val_dataset, learning_rate=1e-4, early_stopping_patience=10, reduce_lr_patience=5):
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )


    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=early_stopping_patience,
        min_delta=1e-4,
        restore_best_weights=True
    )

    log_dir = 'logs/fit/' + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    tensorboard_callback = TensorBoard(
        log_dir=log_dir,
        histogram_freq=1
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=reduce_lr_patience,
        verbose=1,
        min_lr=1e-7
    )

    checkpoint = ModelCheckpoint(
        filepath='CNN_model_weights.weights.h5',
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=True
    )


    history = model.fit(
        x=train_dataset,
        validation_data=val_dataset,
        epochs=1000,
        callbacks=[early_stopping, tensorboard_callback, reduce_lr, checkpoint]
    )
    return history


def evaluate(model, dataset, class_names):
    y_true_onehot = np.concatenate([y for _, y in dataset])
    y_true_indices = np.argmax(y_true_onehot, axis=1)
    y_true = [class_names[i] for i in y_true_indices]

    preds = model.predict(dataset, verbose=0)
    y_pred_indices = np.argmax(preds, axis=1)
    y_pred = [class_names[i] for i in y_pred_indices]

    y_true_group = []
    y_pred_group = []

    start = 0
    for class_idx, class_name in enumerate(class_names):
        count = np.sum(y_true_indices == class_idx)
        class_preds = preds[start:start + count]
        mean_probs = np.mean(class_preds, axis=0)
        group_pred_idx = np.argmax(mean_probs)

        y_true_group.append(class_name)
        y_pred_group.append(class_names[group_pred_idx])

        start += count

    return accuracy_score(y_true, y_pred), accuracy_score(y_true_group, y_pred_group)


if __name__ == '__main__':
    train_dir = '../../PlantCLEF2024singleplanttrainingdata_800_max_side_size_unique_split/train'
    val_dir = '../../PlantCLEF2024singleplanttrainingdata_800_max_side_size_unique_split/val'
    test_dir = '../../PlantCLEF2024singleplanttrainingdata_800_max_side_size_unique_split/test'
    train_dataset, class_names = build_dataset(train_dir)
    val_dataset, _ = build_dataset(val_dir)
    test_dataset, _ = build_dataset(test_dir)
    num_classes = len(class_names)
    model = build_model(num_classes)
    train(model, train_dataset, val_dataset)
    acc, acc_group = evaluate(model, test_dataset, class_names)
    print(f'\n\nPer image accuracy: {acc * 100:.2f}%')
    print(f'Per group accuracy: {acc_group * 100:.2f}%')
