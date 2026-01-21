import os

def run_task(data_dir: str, worker_id: str):
    word_counts = {}
    
    if os.path.exists(data_dir):
        for filename in os.listdir(data_dir):
            if filename.endswith('.txt'):
                with open(os.path.join(data_dir, filename), 'r') as f:
                    for line in f:
                        for word in line.strip().lower().split():
                            word = ''.join(c for c in word if c.isalnum())
                            if word:
                                word_counts[word] = word_counts.get(word, 0) + 1
    
    return list(word_counts.items())
