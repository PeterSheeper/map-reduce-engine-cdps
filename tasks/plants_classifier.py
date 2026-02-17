import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from sklearn.linear_model import LogisticRegression
from tensorflow.data import Dataset
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ConvNeXtTiny
from tensorflow.keras.applications.convnext import preprocess_input
from tensorflow.keras.utils import img_to_array
from tensorflow.keras.utils import to_categorical
from tensorflow import numpy_function

NUM_WORKERS = 3
WORK_DIVISION = None

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
CHUNK_SIZE = 100000
DTYPE_DICT = {
    'image_name': str,
    'organ': str,
    'species_id': int,
    'obs_id': 'Int64',
    'license': str,
    'partner': str,
    'author': str,
    'altitude': float,
    'latitude': float,
    'longitude': float,
    'gbif_species_id': float,
    'species': str,
    'genus': str,
    'family': str,
    'dataset': str,
    'publisher': str,
    'references': str,
    'url': str,
    'learn_tag': str,
    'image_backup_url': str
}


def get_species(data_dir, filename):
    file_path = Path(data_dir) / filename
    species = set()
    chunks = pd.read_csv(file_path, usecols=['species_id'], sep=';', chunksize=CHUNK_SIZE, dtype=DTYPE_DICT)
    for chunk in chunks:
        species.update(chunk['species_id'].dropna().unique())
    return tuple(sorted(species))


def divide_work(classes, num_workers):
    all_pairs = []
    n = len(classes)
    for i in range(n):
        for j in range(i + 1, n):
            all_pairs.append(f'{classes[i]}_{classes[j]}')

    total_pairs = len(all_pairs)
    pairs_per_group = math.ceil(total_pairs / num_workers)
    division = {}
    available_pairs = all_pairs.copy()

    for i in range(num_workers):
        if not available_pairs:
            break
        current_group = []
        first_pair = available_pairs.pop(0)
        current_group.append(first_pair)
        used_in_group = set(first_pair.split('_'))

        while len(current_group) < pairs_per_group and available_pairs:
            best_index = -1
            max_common = -1

            for idx, pair in enumerate(available_pairs):
                pair_classes = set(pair.split('_'))
                common_count = len(pair_classes.intersection(used_in_group))
                if common_count > max_common:
                    max_common = common_count
                    best_index = idx
                if max_common == 2:
                    break

            selected_pair = available_pairs.pop(best_index)
            current_group.append(selected_pair)
            used_in_group.update(selected_pair.split('_'))

        worker = f'worker-{i+1}'
        division[worker] = current_group
    return division


def build_feature_extractor():
    base_model = ConvNeXtTiny(
        include_top=False,
        weights='imagenet',
        input_shape=(*IMG_SIZE, 3)
    )

    model = models.Sequential([
        layers.Input(shape=(*IMG_SIZE, 3)),
        layers.Lambda(preprocess_input),
        base_model,
        layers.GlobalAveragePooling2D()
    ])
    model.trainable = False
    return model


def load_image(path):
    path = path.decode('utf-8')
    try:
        with Image.open(path) as img:
            img = img.convert('RGB')
            w, h = img.size
            if w != h:
                new_size = max(w, h)
                x = (new_size - w) // 2
                y = (new_size - h) // 2
                new_img = Image.new('RGB', (new_size, new_size), (255, 255, 255))
                new_img.paste(img, (x, y))
                img = new_img
            img = img.resize(IMG_SIZE, Image.LANCZOS)
            return img_to_array(img)
    except Exception as e:
        print(f'Error processing {path}: {e}')


def preprocess_path(path, label):
    img = numpy_function(load_image, [path], tf.float32)
    img.set_shape((*IMG_SIZE, 3))
    return img, label


def build_dataset(folder):
    folder = Path(folder)
    class_names = sorted([d.name for d in folder.iterdir() if d.is_dir()])
    class_to_idx = {name: i for i, name in enumerate(class_names)}
    paths = []
    labels = []

    for class_name in class_names:
        class_dir = folder / class_name
        for p in class_dir.iterdir():
            if p.suffix.lower() in {'.jpg', '.jpeg', '.png'}:
                paths.append(str(p))
                labels.append(class_to_idx[class_name])

    paths = np.array(paths)
    labels = to_categorical(labels, num_classes=len(class_names))

    ds = Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(preprocess_path, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds, class_names


def calculate_clf(filename, sid1_list, sid2_list):
    X1 = np.array(sid1_list)
    X2 = np.array(sid2_list)
    X = np.vstack([X1, X2])
    y = np.hstack([
        np.zeros(len(X1), dtype=int),
        np.ones(len(X2), dtype=int)
    ])

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X, y)

    filepath = Path(f'{filename}.joblib')
    joblib.dump(clf, filepath)
    return filepath.stat().st_size


def init_func(worker_id, master_url, data_dir, advertise_host, port):
    """Makes init operations based on worker data."""
    global WORK_DIVISION
    species = get_species(data_dir, 'PlantCLEF2024singleplanttrainingdata_30000.csv')
    WORK_DIVISION = divide_work(species, NUM_WORKERS)


def map_func(data_dir, worker_id):
    """Returns list of (key, value) pairs."""
    results = []
    model = build_feature_extractor()
    ds, class_names = build_dataset(data_dir)
    embeddings = model.predict(ds, verbose=0)
    y_onehot = np.concatenate([y for _, y in ds])
    y_indices = np.argmax(y_onehot, axis=1)

    for embedding, y_idx in zip(embeddings, y_indices):
        sid = class_names[y_idx]
        results.append((sid, embedding.tolist()))

    return results


def shuffle_func(key):
    """
    Decides which worker gets this key.
    Returns a list of integers, for each integer engine does: target % num_workers
    """
    workers = []
    for worker_id, values in WORK_DIVISION.items():
        for v in values:
            sid1, sid2 = v.split('_')
            if key == sid1 or key == sid2:
                idx = int(''.join([c for c in worker_id if c.isdigit()]))
                workers.append(idx-1)
                break
    return workers


def reduce_func(data, worker_id):
    """Processes data (in format: List[dict_item[key, values]]."""
    results = []
    data_lookup = {key: values for key, values in data}
    assigned_pairs = WORK_DIVISION[worker_id]
    for filename in assigned_pairs:
        sid1, sid2 = filename.split('_')
        file_size = calculate_clf(filename, data_lookup[sid1], data_lookup[sid2])
        results.append(f'Calculated classifier {filename}: {file_size} bytes')
    return results
