import os
import numpy as np
import shutil
import itertools

def run(master_path = 'E:\\csv_from_server\\2022_서울시\\4\\csv_result'):
    files = os.listdir(master_path)
    divided = [(f.split('_')) for f in files]
    new = list(itertools.chain.from_iterable(divided))
    unique = set(new)
    droped = [f for f in unique if not ((f == "box") or ("csv" in f))]
    for d in droped:
        for (path, dir, files) in os.walk(master_path):
            for f in files:
                if d in f:
                    os.makedirs(os.path.join(master_path, d), exist_ok=True)
                    shutil.move(os.path.join(path, f), os.path.join(master_path, d, f))


if __name__ == "__main__":
    m_path = input()
    run(master_path = m_path)