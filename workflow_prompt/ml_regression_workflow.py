class Node:
    def __init__(self, name):
        self.name = name
        self.children = []

    def add_child(self, child):
        self.children.append(child)

def print_workflow(node, indent=""):
    print(f"{indent}- {node.name}")
    for child in node.children:
        print_workflow(child, indent + "  ")

# Define the workflow nodes
root = Node("Workflow")
task_a = Node("Task A")
task_b = Node("Task B")
task_c = Node("Task C")
task_d = Node("Task D")
task_e = Node("Task E")

# Build the tree structure
root.add_child(task_a)
root.add_child(task_b)
task_a.add_child(task_c)
task_a.add_child(task_d)
task_b.add_child(task_e)

# Display the workflow tree
print_workflow(root)