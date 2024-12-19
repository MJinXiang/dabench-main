

from da_agent.agent.action import Bash, Action, Terminate, Python, SQL, ListFiles, LLMQuery, CheckOutputWithLLM, AddNewToolAction, QueryToolsAction

class WorkflowNode:
    def __init__(self, action, next_nodes=None):
        self.action = action          # 该节点对应的动作
        self.next_nodes = next_nodes or []  # 后续可能的节点列表

    def add_next_node(self, node):
        self.next_nodes.append(node)

    def remove_next_node(self, node):
        self.next_nodes.remove(node)

# 定义动作
# start_action = ListFiles
process_action = Python
start_action = AddNewToolAction
query_tools_action = QueryToolsAction
# verify_action = CheckOutputWithLLM
# sub_process_action = Bash
end_action = Terminate

# 创建节点
start_node = WorkflowNode(action=start_action)
process_node = WorkflowNode(action=process_action)
# add_tool_node = WorkflowNode(action=add_tool_action)
query_tools_node = WorkflowNode(action=query_tools_action)
# verify_node = WorkflowNode(action=verify_action)
# sub_process_node = WorkflowNode(action=sub_process_action)
end_node = WorkflowNode(action=end_action)

# 建立节点关系
# start_node.next_nodes = [process_node]
# # process_node.next_nodes = [verify_node]
# # verify_node.next_nodes = [sub_process_node]  
# process_node.next_nodes = [sub_process_node]
# # sub_process_node.next_nodes = [process_node, end_node]  # 根据情况回到 process_node 或结束
# sub_process_node.next_nodes = [end_node]

start_node.next_nodes = [process_node]
process_node.next_nodes = [query_tools_node]
# add_tool_node.next_nodes = [query_tools_node]
query_tools_node.next_nodes = [end_node]
# sub_process_node.next_nodes = [end_node]

#开始节点
workflow_start_node = start_node


def get_all_actions(start_node):
    actions = set()
    nodes = [start_node]
    while nodes:
        current = nodes.pop()
        actions.add(current.action)
        nodes.extend(current.next_nodes)
    return list(actions)

# 导出可用的动作类
AVAILABLE_ACTION_CLASSES = get_all_actions(workflow_start_node)



# workflow = [
#     ListFiles,
#     Python,
#     CheckOutputWithLLM,
#     Terminate
# ]



# 定义工作流程
# workflow = [
#     ListFiles(directory="path/to/directory"),
#     Python(file_path="path/to/python_file.py"),
#     CheckOutputWithLLM(output="output files or content", target="target file or task description"),
#     Terminate(output="literal_answer_or_output_path")
# ]

# # 代理的执行循环
# for action in workflow:
#     # 可在这里添加每个动作执行前的日志或检查
#     print(f"Executing action: {action}")
#     result = action.execute()
#     # 可在这里处理执行结果或根据需要调整下一步
#     if isinstance(action, Terminate):
#         print("Workflow terminated.")
#         break