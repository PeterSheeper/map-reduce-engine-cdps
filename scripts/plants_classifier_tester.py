from pathlib import Path

import joblib
from sklearn.metrics import accuracy_score
import numpy as np

from tasks.plants_classifier import build_dataset
from tasks.plants_classifier import build_feature_extractor


def load_classifiers(clf_dir):
    clf_dir = Path(clf_dir)
    classifiers = {}
    for file_path in clf_dir.glob('*.joblib'):
        sid1, sid2 = file_path.stem.split('_')
        classifiers[(sid1, sid2)] = joblib.load(file_path)
    return classifiers


def predict(embeddings, classifiers):
    num_samples = embeddings.shape[0]
    votes = [{} for _ in range(num_samples)]
    for (sid1, sid2), clf in classifiers.items():
        preds = clf.predict(embeddings)
        for i, pred in enumerate(preds):
            if pred == 0:
                if not sid1 in votes[i]:
                    votes[i][sid1] = 0
                votes[i][sid1] += 1
            else:
                if not sid2 in votes[i]:
                    votes[i][sid2] = 0
                votes[i][sid2] += 1
    return [max(v.items(), key=lambda x: x[1])[0] for v in votes]


def evaluate(test_dir, clf_dir, feature_extractor):
    classifiers = load_classifiers(clf_dir)
    ds, class_names = build_dataset(test_dir)
    embeddings = feature_extractor.predict(ds)

    y_pred = predict(embeddings, classifiers)

    y_true_onehot = np.concatenate([y for _, y in ds])
    y_true_indices = np.argmax(y_true_onehot, axis=1)
    y_true = [class_names[i] for i in y_true_indices]

    y_true_group = []
    y_pred_group = []

    for class_name in class_names:
        mask = np.array(y_true) == class_name
        class_preds = np.array(y_pred)[mask]
        group_pred = max(set(class_preds), key=lambda pred: np.sum(class_preds == pred))
        y_pred_group.append(group_pred)
        y_true_group.append(class_name)

    return accuracy_score(y_true, y_pred), accuracy_score(y_true_group, y_pred_group)


if __name__ == "__main__":
    feature_extractor = build_feature_extractor()
    test_dir = '../../PlantCLEF2024singleplanttrainingdata_800_max_side_size_10000_split/test'
    clf_dir = '../../OvO_10000'
    acc, acc_group = evaluate(test_dir, clf_dir, feature_extractor)
    print(f'\n\nPer image accuracy: {acc * 100:.2f}%')
    print(f'Per group accuracy: {acc_group * 100:.2f}%')
