import os
import zlib


def map_func(data_dir, worker_id):
    """Returns list of (key, value) pairs."""
    results = []

    if os.path.exists(data_dir):
        for filename in os.listdir(data_dir):
            if filename.endswith('.txt'):
                with open(os.path.join(data_dir, filename), 'r') as f:
                    for line in f:
                        for word in line.strip().lower().split():
                            word = ''.join(c for c in word if c.isalnum())
                            if word:
                                results.append((word, 1))

    return results


def shuffle_func(key):
    """
    Decides which worker gets this key.
    Returns a list of integers, for each integer engine does: target % num_workers
    """
    return [zlib.adler32(str(key).encode())]


def reduce_func(data, worker_id):
    """Processes data (in format: List[dict_item[key, values]]."""
    results = []
    for key, values in data:
        results.append((key, sum(values)))
    return results
