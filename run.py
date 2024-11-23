import argparse
import datetime
import json
import logging
import os
import random
import sys

from tqdm import tqdm

from da_agent.envs.da_agent import DA_Agent_Env
from da_agent.agent.agents import PromptAgent
# from da_agent.agent.workflow import workflow


#  Logger Configs {{{ #
logger = logging.getLogger("da_agent")
logger.setLevel(logging.DEBUG)

datetime_str: str = datetime.datetime.now().strftime("%Y%m%d@%H%M%S")

file_handler = logging.FileHandler(os.path.join("logs", "normal-{:}.log".format(datetime_str)), encoding="utf-8")
debug_handler = logging.FileHandler(os.path.join("logs", "debug-{:}.log".format(datetime_str)), encoding="utf-8")
stdout_handler = logging.StreamHandler(sys.stdout)
sdebug_handler = logging.FileHandler(os.path.join("logs", "sdebug-{:}.log".format(datetime_str)), encoding="utf-8")

file_handler.setLevel(logging.INFO)
debug_handler.setLevel(logging.DEBUG)
stdout_handler.setLevel(logging.INFO)
sdebug_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter(
    fmt="\x1b[1;33m[%(asctime)s \x1b[31m%(levelname)s \x1b[32m%(module)s/%(lineno)d-%(processName)s\x1b[1;33m] \x1b[0m%(message)s")
file_handler.setFormatter(formatter)
debug_handler.setFormatter(formatter)
stdout_handler.setFormatter(formatter)
sdebug_handler.setFormatter(formatter)

stdout_handler.addFilter(logging.Filter("da_agent"))
sdebug_handler.addFilter(logging.Filter("da_agent"))

logger.addHandler(file_handler)
logger.addHandler(debug_handler)
logger.addHandler(stdout_handler)
logger.addHandler(sdebug_handler)
#  }}} Logger Configs # 



def config() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run end-to-end evaluation on the benchmark"
    )
    
    parser.add_argument("--max_steps", type=int, default=20)
    
    parser.add_argument("--max_memory_length", type=int, default=15)
    parser.add_argument("--suffix", '-s', type=str, default="")
    
    parser.add_argument("--model",'-m',type=str, default="gpt-4o")  #更改模型
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_tokens", type=int, default=1500)
    parser.add_argument("--stop_token", type=str, default=None)
    
    # example config
    parser.add_argument("--workflow_dir", type=str, default="workflow_prompt/ml_cluster_workflow.txt",help="Path to the workflow file.")
    parser.add_argument("--task_config","-t", type=str, default="da_code/configs/task/examples copy 2.jsonl")
    parser.add_argument("--source_dir", type=str, default="da_code/source")
    parser.add_argument("--example_index", "-i", type=str, default="all", help="index range of the examples to run, e.g., '0-10', '2,3', 'all'")
    parser.add_argument("--example_name", "-n", type=str, default="", help="name of the example to run")
    parser.add_argument("--overwriting", action="store_true", default=True)
    parser.add_argument("--retry_failed", action="store_true", default=False)

    # output related
    parser.add_argument("--output_dir", type=str, default="output")
    args = parser.parse_args()

    return args


def test(
    args: argparse.Namespace,
    test_all_meta: dict = None
) -> None:
    scores = []
    
    # log args
    logger.info("Args: %s", args)

    if args.suffix == "":
        logger.warning("No suffix is provided, the experiment id will be the model name.")
        experiment_id = args.model.split("/")[-1]
    else:
        experiment_id = args.model.split("/")[-1] + "-" + args.suffix

    env_config = \
    {
        "image_name": "da_agent-image",
        "init_args": {
            "name": experiment_id,
            "work_dir": "/workspace",
        }
    }
    
    agent = PromptAgent(
        model=args.model,
        max_tokens=args.max_tokens,
        top_p=args.top_p,
        temperature=args.temperature,
        max_memory_length=args.max_memory_length,
        max_steps=args.max_steps,
    )
    
    ## load task configs
    assert os.path.exists(args.task_config) and args.task_config.endswith(".jsonl"), f"Invalid task_config, must be a valid jsonl file: {args.task_config}"
    with open(args.task_config, "r", encoding='utf-8') as f:
        task_configs = [json.loads(line) for line in f]
    #只跑固定的例子，以id或索引为准
    if args.example_name != "":
        task_configs = [task for task in task_configs if args.example_name in task["id"]]
    else:
        if args.example_index != "all":
            if "-" in args.example_index:
                start, end = map(int, args.example_index.split("-"))
                task_configs = task_configs[start:end]
            else:
                indices = list(map(int, args.example_index.split(",")))
                task_configs = [task_configs[i] for i in indices]

    with open(args.workflow_dir, "r",encoding="utf-8") as f:
        workflow_config = f.read()

    #遍历task
    for task_config in task_configs:
        instance_id = experiment_id +"/"+ task_config["id"] #模型名/任务id
        output_dir = os.path.join(args.output_dir, instance_id) #共同组成输出路径
        result_json_path =os.path.join(output_dir, "dabench/result.json") #json输出路径
        #overwriting为true的时候，已经存在的文件不会被覆盖，为false的时候会覆盖，默认为false
        if not args.overwriting and os.path.exists(result_json_path):
            logger.info("Skipping %s", instance_id)
            continue
        elif os.path.exists(result_json_path):
            logger.info("Overwriting %s", instance_id)
        else:
            logger.info("Running %s", instance_id)
        #检查是否有需要重试的例子，如要重试，将retry_failed设为true，但仅仅是记录日志
        if args.retry_failed and os.path.exists(result_json_path):
            with open(result_json_path, "r") as f:
                result = json.load(f)
                #如果已经完成，且没有error或FAIL，就跳过
                if result["finished"] and (not "FAIL" in result["result"]) and (not "error" in result["result"].lower()):
                    logger.info("Skipping %s", instance_id)
                    continue
            logger.info("Retrying %s", instance_id)
        #确保没有output路径，有的话就删掉
        if os.path.exists(output_dir):
            os.system(f"rm -rf {output_dir}")
            logger.info("Removed existing %s", output_dir)

        os.makedirs(output_dir, exist_ok=True)

        env_config["init_args"]["name"] = experiment_id +"-"+ task_config["id"]
        env = DA_Agent_Env(
            env_config=env_config,
            task_config=task_config,
            source_dir=args.source_dir,
            # workflow=workflow,
            cache_dir="./cache", #缓存目录
            mnt_dir=output_dir #输出目录
        )
    
        agent.set_env_and_task(env)
    
        logger.info('Task input:' + task_config['instruction'])
        done, result_output = agent.run()
        trajectory = agent.get_trajectory()

        os.makedirs(os.path.join(output_dir, "dabench"), exist_ok=True)
        result_files = env.post_process()
        dabench_result = {"finished": done, "steps": len(trajectory["trajectory"]),
                           "result": result_output,"result_files": result_files, **trajectory}
        with open(os.path.join(output_dir, "dabench/result.json"), "w") as f:
            json.dump(dabench_result, f, indent=2)
        
        logger.info("Finished %s", instance_id)
        env.close()



if __name__ == '__main__':
    args = config()
    
    test(args)