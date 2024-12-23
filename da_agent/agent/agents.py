import base64
import json
import logging
import ast
import os
from glob import glob
import re
import time
import uuid
from http import HTTPStatus
from io import BytesIO
from typing import Dict, List
from da_agent.agent.prompts import SYS_PROMPT_IN_OUR_CODE, SYS_PROMPT_PLOT_BAR
from da_agent.agent.COT_prompts import ELABORATE_DEEP_THINK_PROMPT_CODING
from da_agent.agent.action import Bash, Action, Terminate, Python, SQL, ListFiles, LLMQuery, CheckOutputWithLLM,AddNewToolAction, QueryToolsAction
from da_agent.envs.da_agent import DA_Agent_Env
from openai import AzureOpenAI
from typing import Dict, List, Optional, Tuple, Any, TypedDict
from transformers.agents.agents import Toolbox
# agents.py
# from da_agent.agent.workflow import WorkflowNode, workflow_start_node
from da_agent.agent.models import call_llm
from da_agent.agent.workflow import AVAILABLE_ACTION_CLASSES
from da_agent.agent.generatedtool import GeneratedTool, add_parent_pointers, parse_generated_tools
from da_agent.agent.tool_retriever import ToolRetrievalTool



MAX_OBSERVATION_LENGTH = 2000
TIME_OUT_ACTION = 600

logger = logging.getLogger("da_agent")


class PromptAgent:
    def __init__(
        self,
        model="gpt-4o",
        max_tokens=1500,
        top_p=0.9,
        temperature=0.5,
        max_memory_length=10,
        max_steps=15,
        #新增
        generated_tool_dir="",
        disable_accum=False,
        *args,
        **kwargs,
    ):
        
        self.model = model
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.temperature = temperature
        self.max_memory_length = max_memory_length
        self.max_steps = max_steps

        # 动态动作生成相关的初始化
        self.generated_tool_dir = generated_tool_dir
        self.disable_accum = disable_accum
        # 加载生成的工具
        self.generated_toolbox = self.load_generated_tools()
        # 用于追踪函数调用次数（可选）
        self.prev_num_calls = {}
   
        # 初始化 ToolRetrievalTool
        self.tool_retrieval_tool = ToolRetrievalTool(self.generated_tool_dir)
        # 将 ToolRetrievalTool 添加到工具箱
        self.generated_toolbox.add_tool(self.tool_retrieval_tool)
        
        
        self.thoughts = []
        self.responses = []
        self.actions = []
        self.observations = []
        self.system_message = ""
        self.history_messages = []
        self.env = None
        self.codes = []
        # self._AVAILABLE_ACTION_CLASSES = [Bash, Python, SQL, Terminate, ListFiles, LLMQuery, CheckOutputWithLLM]
        self._AVAILABLE_ACTION_CLASSES = AVAILABLE_ACTION_CLASSES
        self.work_dir = "/workspace"
        
    def set_env_and_task(self, env: DA_Agent_Env):
        self.env = env
        self.thoughts = []
        self.responses = []
        self.actions = []
        self.observations = []
        self.codes = []
        self.history_messages = []
        self.instruction = self.env.task_config['instruction']

        # 加载生成的工具
        self.generated_toolbox = self.load_generated_tools()
        
        # self.workflow = self.env.workflow
        # 设置起始节点
        self.workflow_start_node = self.env.workflow_start_node
        action_space = "".join([action_cls.get_action_description() for action_cls in self._AVAILABLE_ACTION_CLASSES])
        # self.system_message = SYS_PROMPT_IN_OUR_CODE.format(work_dir=self.work_dir, action_space=action_space, task=self.instruction, max_steps=self.max_steps,workflow=self.workflow)
        self.system_message = SYS_PROMPT_PLOT_BAR.format(work_dir=self.work_dir, action_space=action_space, task=self.instruction, max_steps=self.max_steps)
        self.history_messages.append({
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": self.system_message 
                },
            ]
        })
        
    def predict(self, obs: Dict=None) -> List:
        """
        Predict the next action(s) based on the current observation.
        """    
        #确保observation=actions=thoughts
        assert len(self.observations) == len(self.actions) and len(self.actions) == len(self.thoughts) \
            , "The number of observations and actions should be the same."

        status = False
        while not status:
            #将一个新的用户消息添加到消息历史记录中
            messages = self.history_messages.copy()

            messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Observation: {}\n".format(str(obs)) #现有观察
                    }
                ]
            })  

            status, response = call_llm({
                "model": self.model,
                "messages": messages,
                "max_tokens": self.max_tokens,
                "top_p": self.top_p,
                "temperature": self.temperature
            })
            response = response.strip()
            #保留第一个消息，然后从第四个消息开始保留所有后续消息。这样可以减少历史消息的数量，以应对上下文长度超过限制或令牌数超过限制的情况。
            if not status:
                if response in ["context_length_exceeded","rate_limit_exceeded","max_tokens"]:
                    self.history_messages = [self.history_messages[0]] + self.history_messages[3:]
                else:
                    raise Exception(f"Failed to call LLM, response: {response}")
            
        try:
            #提取action
            action = self.parse_action(response)

            #提取thought
            thought = re.search(r'Thought:(.*?)Action', response, flags=re.DOTALL)
            if thought:
                thought = thought.group(1).strip()
            else:
                thought = response
        except ValueError as e:
            print("Failed to parse action from response", e)
            action = None
        
        logger.info("Observation: %s", obs)
        logger.info("Response: %s", response)

        self._add_message(obs, thought, action)
        self.observations.append(obs)
        self.thoughts.append(thought)
        self.responses.append(response)
        self.actions.append(action)

        if action is not None:
             # 保存新生成的工具（如果有）
            if isinstance(action, AddNewToolAction) and not self.disable_accum:
                self.save_generated_tools(action.code)

            self.codes.append(action.code)
        else:
            self.codes.append(None)
       
        return response, action
    


    def _add_message(self, observations: str, thought: str, action: Action):
        self.history_messages.append({
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Observation: {}".format(observations)
                }
            ]
        })
        self.history_messages.append({
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": "Thought: {}\n\nAction: {}".format(thought, str(action))
                }
            ]
        })
        #防止过最大记忆量，从中删除前面的
        if len(self.history_messages) > self.max_memory_length*2+1:
            self.history_messages = [self.history_messages[0]] + self.history_messages[-self.max_memory_length*2:]
    
    def parse_action(self, output: str) -> Action:
        """ Parse action from text """
        if output is None or len(output) == 0:
            pass
        action_string = ""
        patterns = [r'["\']?Action["\']?:? (.*?)Observation',r'["\']?Action["\']?:? (.*?)Thought', r'["\']?Action["\']?:? (.*?)$', r'^(.*?)Observation']

        for p in patterns:
            match = re.search(p, output, flags=re.DOTALL)
            if match:
                action_string = match.group(1).strip()
                break
        if action_string == "":
            action_string = output.strip()
        
        output_action = None
        for action_cls in self._AVAILABLE_ACTION_CLASSES:
            action = action_cls.parse_action_from_text(action_string)
            if action is not None:
                output_action = action
                break
        if output_action is None:
            action_string = action_string.replace("\_", "_").replace("'''","```")
            for action_cls in self._AVAILABLE_ACTION_CLASSES:
                action = action_cls.parse_action_from_text(action_string)
                if action is not None:
                    output_action = action
                    break
        
        return output_action
    

     #检查函数名是否冲突
    def check_collision(self, code_action: str):
        # Make sure code_action has no syntax errors first
        try:
            tree = ast.parse(code_action)
        except:
            return

        add_parent_pointers(tree)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and isinstance(node.parent, ast.Module):
                name = node.name
                if name in self.generated_toolbox.tools:
                    if name not in self.metrics["collision"]:
                        self.metrics["collision"][name] = 1
                    else:
                        self.metrics["collision"][name] += 1
                    # error_msg = f"Function name '{name}' already exists. Please choose a different name."
                    # raise AgentExecutionError(error_msg)

    def add_decorators(self, code_action: str) -> str:
        # TODO: Need to add decorator to generated functions that were loaded from disk as well
        try:
            tree = ast.parse(code_action)
            add_parent_pointers(tree)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and isinstance(node.parent, ast.Module):
                    decorator = ast.Name(id="track_num_calls", ctx=ast.Load())
                    node.decorator_list.append(decorator)
            updated_code_action = ast.unparse(tree)
            return updated_code_action
        except:
            print("Add decorator failed :( returning original code_action")
            return code_action

    def load_generated_tools(self):
        generated_tools: list[GeneratedTool] = []
        generated_tool_paths = sorted(glob(os.path.join(self.generated_tool_dir, "*.py")))
        for path in generated_tool_paths:
            with open(path, "r") as f:
                code = f.read()
            tools = parse_generated_tools(code)
            generated_tools.extend(tools)
            # 将生成的工具加载到环境中
            # self.env.step(code)
        return Toolbox(generated_tools)
   

    def remove_shell_commands(self, code_action: str) -> str:
        shell_cmds = []
        no_cmds_code_action = []
        for line in code_action.split("\n"):
            if line.startswith("!"):
                shell_cmds.append(line)
            else:
                no_cmds_code_action.append(line)

        shell_cmds = "\n".join(shell_cmds)
        code_action = "\n".join(no_cmds_code_action)
        return shell_cmds, code_action
    
    def save_generated_tools(self, code_action: str):
        # _, code_action = self.remove_shell_commands(code_action)
        # print(f"Saving generated tool:\n{code_action}")
        generated_tools = parse_generated_tools(code_action)

        for tool in generated_tools:
            if tool.name in self.generated_toolbox.tools:
                print(f"Tool '{tool.name}' already exists. Removing it before adding the new one.")
                self.generated_toolbox.remove_tool(tool.name)

            self.generated_toolbox.add_tool(tool)

            tool_id = len(self.generated_toolbox.tools)
            file_name = f"{str(tool_id).zfill(4)}_{tool.name}.py"
            file_path = os.path.join(self.generated_tool_dir, file_name)

            content = tool.code
            if tool.dependencies:
                content = f"{tool.dependencies}\n\n\n{content}"

            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)

    def prerun(self, code_action: str) -> str:
        shell_cmds, code_action = self.remove_shell_commands(code_action)
        code_action = self.correct_docstring(code_action)
        self.check_collision(code_action)
        code_action = self.add_decorators(code_action)

        # Add the shell_commands back
        code_action = shell_cmds + "\n" + code_action
        return code_action

    def correct_docstring(self, code_action: str) -> str:
        try:
            tree = ast.parse(code_action)
        except Exception as e:
            print(f"Attempt to correct docstring failed due to the following error: {e}")
            return code_action

        add_parent_pointers(tree)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and isinstance(node.parent, ast.Module):
                if ast.get_docstring(node) is None:
                    func = ast.unparse(node)
                    messages = [
                        {
                            "role": "user",
                            "content": f"Write a one-line docstring for the following Python function:\n```\n{func}\n```",
                        }
                    ]
                    # resp = call_llm(messages) #调了api，写文档字符串，用来描述当前action的作用
                    success, resp = call_llm({
                                        "model": self.model,
                                        "messages": messages,
                                        "max_tokens": self.max_tokens,
                                        "top_p": self.top_p,
                                        "temperature": self.temperature
                                    })
                    if success:
                        try:
                            docstring = re.findall(r'"""(.*?)"""', resp, re.DOTALL)[0]
                            node.body.insert(0, ast.Expr(value=ast.Constant(value=docstring))) #将提取到的文档字符串插入到函数定义的AST节点 node 的主体部分的第一个位置，形成标准的文档字符串。
                        except Exception as e:
                            print(f"Attempt to correct docstring failed due to the following error: {e}")
                            return code_action
                    else:
                        print(f"LLM call failed. Unable to generate docstring for the function: {node.name}")
                        return code_action

        try:
            corrected_code_action = ast.unparse(tree)
            return corrected_code_action
        except Exception as e:
            print(f"Attempt to correct docstring failed due to the following error: {e}")
            return code_action



    def run(self):
        assert self.env is not None, "Environment is not set."
        result = ""
        done = False
        step_idx = 0
        obs = "You are in the folder now."
        # workflow_step = 0  # 新增：工作流步骤指针
        retry_count = 0
        last_action = None
        repeat_action = False

        # 设置当前工作流节点为起始节点
        current_node = self.workflow_start_node
        while not done and step_idx < self.max_steps:
            
            if current_node:
                # 将当前步骤的动作信息加入到 obs 中
                obs_with_workflow = f"{obs}\n\nYou can proceed with the following Action:\n{current_node.action.get_this_action()}"
            else:
                # 如果没有当前节点，工作流可能已经结束
                obs_with_workflow = obs

            _, action = self.predict(
                # obs
                obs_with_workflow
            )
            if action is None:
                logger.info("Failed to parse action from response, try again.")
                retry_count += 1
                if retry_count > 3:
                    logger.info("Failed to parse action from response, stop.")
                    break
                obs = "Failed to parse action from your response, make sure you provide a valid action."
            else:
                logger.info("Step %d: %s", step_idx + 1, action)
                #防止action重复
                if last_action is not None and last_action == action:
                    if repeat_action:
                        return False, "ERROR: Repeated action"
                    else:
                        obs = "The action is the same as the last one, please provide a different action."
                        repeat_action = True
                else:
                    obs, done = self.env.step(action)
                    last_action = action
                    repeat_action = False

                    # # 如果有工作流，检查动作是否与当前工作流步骤匹配
                    # if self.workflow and workflow_step < len(self.workflow):
                    #     expected_action = self.workflow[workflow_step]
                    #     if not isinstance(action, expected_action):
                    #         # 如果不匹配，可以给 LLM 提示
                    #         obs += f"\nNote: Your action '{action}' does not match the expected step '{expected_action}'."
                    #     else:
                    #         workflow_step += 1

                    # 检查动作是否与当前节点的动作匹配
                    if not isinstance(action, current_node.action):
                        obs += f"\nNote: Your action does not match the expected action '{current_node.action}'."
                        # 保持在当前节点，不前进
                    else:
                        # 动作匹配，前进到下一个节点
                        if len(current_node.next_nodes) == 1:
                            current_node = current_node.next_nodes[0]
                        elif len(current_node.next_nodes) > 1:
                            # 根据业务逻辑选择下一个节点
                            selected_node = self.choose_next_node(current_node.next_nodes, action, obs)
                            current_node = selected_node
                        else:
                            # 没有后续节点，结束工作流
                            current_node = None

            if done:
                if isinstance(action, Terminate):
                    result = action.output
                logger.info("The task is done.")
                break
            step_idx += 1

        return done, result
    

    def call_llm_evaluate(self, obs, node_instructions):
        prompt = (
            f"Based on the following observations, evaluate the suitability of each possible action step and provide a score (0 to 100).\n\n"
            f"Observation: {obs}\n\n"
            f"Available actions:\n"
            f"Notes:\n1. Please consider the context and the current state of the system.\n2. Higher scores indicate better suitability.\n3. If the action is not suitable, please provide a reason.\n"
        )
        for idx, instruction in enumerate(node_instructions):
            prompt += f"{idx + 1}. {instruction}\n"
        prompt += "\nPlease provide a score for each action step in the following format:\nAction1: 85\nAction2: 70\n..."

        print(f"evaluate prompt:{prompt}")

        payload = {
            "model": "gpt-4",  
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 500,
            "temperature": 0
        }

        try:
            success, output_message = call_llm(payload)
            if success:
                text = output_message.strip()
                scores = {}

                # Use regular expressions to parse scores
                pattern = r"Action(\d+):\s*(\d+)"
                matches = re.findall(pattern, text)
                for match in matches:
                    try:
                        action_num = int(match[0]) - 1  # Index starts from 0
                        score = float(match[1])
                        scores[action_num] = score
                    except ValueError:
                        print(f"Unable to parse score: Action{match[0]}: {match[1]}")

                if not scores:
                    print("Failed to parse any scores.")

                return scores
            else:
                print(f"LLM evaluation failed: {output_message}")
                return {}
        except Exception as e:
            print(f"Exception occurred while calling `call_llm_evaluate`: {e}")
            return {}

    def choose_next_node(self, next_nodes, action, obs):
        node_instructions = [node.action.get_this_action() for node in next_nodes]
        
        # Call LLM to get scores
        llm_scores = self.call_llm_evaluate(obs, node_instructions)
        
        # Aggregate scores (only using LLM scores)
        total_scores = {}
        for idx in range(len(next_nodes)):
            llm_score = llm_scores.get(idx, 0)
            total_scores[idx] = llm_score  # Only using LLM scores
        
        # Select the node with the highest score
        if total_scores:
            selected_idx = max(total_scores, key=total_scores.get)
            selected_node = next_nodes[selected_idx]
            print(f"Selected node {selected_idx + 1}, Score: {total_scores[selected_idx]}")
            return selected_node
        else:
            # If scoring fails, default to the first node
            print("Failed to obtain scores, defaulting to the first node.")
            return next_nodes[0]


    def get_trajectory(self):
        trajectory = []
        for i in range(len(self.observations)):
            trajectory.append({
                "observation": self.observations[i],
                "thought": self.thoughts[i],
                "action": str(self.actions[i]),
                "code": self.codes[i],
                "response": self.responses[i]
            })
        trajectory_log = {
            "Task": self.instruction, #任务描述
            "system_message": self.system_message, #prompt
            "trajectory": trajectory
        }
        return trajectory_log


if __name__ == "__main__":
    agent = PromptAgent()
    response = """Bash(code=\"\"ls -a\"):\n\n(Note: I am using the 'ls -a' command to list all files, including hidden ones, in the working directory. This will help me ensure that I am in the correct directory and provide a reference for the file paths.\")"""
    import pdb; pdb.set_trace()
    action = agent.parse_action(response)
    print(action)