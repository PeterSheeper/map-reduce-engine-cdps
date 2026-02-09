import math
import pandas as pd
from pathlib import Path

NUM_WORKERS = 3
WORK_DIVISION = None


def get_species(data_dir, filename):
    file_path = Path(data_dir) / filename
    species = set()
    chunks = pd.read_csv(file_path, usecols=['species_id'], chunksize=100000)
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


def calculate_clf(filename, sid1_list, sid2_list):
    file_path = Path(f'{filename}.txt')
    with file_path.open('w', encoding='utf-8') as f:
        f.write("--- SID1 LIST ---\n")
        f.writelines(f"{item}\n" for item in sid1_list)
        f.write("\n--- SID2 LIST ---\n")
        f.writelines(f"{item}\n" for item in sid2_list)
    return file_path.stat().st_size


def init_func(worker_id, master_url, data_dir, advertise_host, port):
    """Makes init operations based on worker data."""
    global WORK_DIVISION
    species = get_species(data_dir, 'PlantCLEF2024singleplanttrainingdata-template.csv')
    WORK_DIVISION = divide_work(species, NUM_WORKERS)


def map_func(data_dir, worker_id):
    """Returns list of (key, value) pairs."""
    results = []
    data_dir = Path(data_dir)
    for sid_dir in data_dir.iterdir():
        if sid_dir.is_dir():
            sid = sid_dir.name
            for f in sid_dir.glob("*.txt"):
                try:
                    content = f.read_text(encoding='utf-8').strip()
                    results.append((sid, content))
                except Exception as e:
                    print(f"Exception for {f}: {e}")
    return results


def shuffle_func(key):
    """Decides which worker gets this key.
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
