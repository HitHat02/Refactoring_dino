import os
import pandas as pd

def run(master_path = 'E:\\csv_from_server\\2022_양주'):
    result_dict = {}
    for (path, dir, files) in os.walk(master_path):
        if len(files) == 0 : continue
        if '.csv' in files[0]:
            print(path, dir, files)
            if files[0] == 'box_total.csv':
                continue
            b_name = os.path.basename(path)
            dfs = [ pd.read_csv(os.path.join(path, f)) for f in files]
            result = pd.concat(dfs, ignore_index=True)
            result_dict[b_name] = result
            result.to_csv(os.path.join(path, f"box_{b_name}_total.csv"), index=False, encoding='utf-8-sig')

    os.makedirs(os.path.join(master_path, 'totals'), exist_ok=True)
    for k, v in result_dict.items():
        v.to_csv(os.path.join(master_path, "totals", f"box_{k}_total.csv"), index=False, encoding='utf-8-sig')

if __name__ == "__main__":
    m_path = input()
    run(master_path = m_path)