import os

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


def partition_func(key):
    """
    Decides which worker gets this key.
    Returns an integer, engine does: partition_func(key) % num_workers
    """
    return hash(str(key))
