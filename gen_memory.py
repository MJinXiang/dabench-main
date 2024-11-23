import json
import logging
import os
import random
import sys
import argparse
import datetime

from tqdm import tqdm


def main(args:argparse.Namespace)-> None:
    logging.info("Args: %s",args)
     # 读取 task_config 文件
    with open(args.task_config, 'r',encoding='utf-8') as f:
        tasks = [json.loads(line) for line in f]

     # 提取所有 id
    ids = [task['id'] for task in tasks]

     # 汇总 trajectory 数据
    trajectories = []

    for task_id in ids:
        task_output_dir = os.path.join(args.output_dir, task_id, 'dabench')
        if os.path.exists(task_output_dir):
            file_path=os.path.join(task_output_dir, 'result.json')

            with open(file_path, 'r') as f:
                data = json.load(f)
            if 'trajectory' in data:
                trajectories.append(data['trajectory'])
            else:
                logging.warning(f"No 'trajectory' field found in {file_path}")
        else:
            logging.warning(f"Directory {task_output_dir} does not exist")

     # 将汇总的 trajectory 数据写入新的 json 文件
    output_file = os.path.join(args.results_dir, 'all_trajectories.json')
    with open(output_file, 'w') as f:
        json.dump(trajectories, f, indent=2)

    logging.info(f"All trajectories have been written to {output_file}")

def config() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    
    # example config
    parser.add_argument("--task_config","-t", type=str, default="da_code/configs/task/examples.jsonl")
    parser.add_argument("--example_index", "-i", type=str, default="all", help="index range of the examples to run, e.g., '0-10', '2,3', 'all'")
    parser.add_argument("--example_name", "-n", type=str, default="", help="name of the example to run")
   
    # output related
    parser.add_argument("--output_dir", type=str, default="output/gpt-4o")
    parser.add_argument("--results_dir", type=str, default="output")
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args=config()
    main(args)