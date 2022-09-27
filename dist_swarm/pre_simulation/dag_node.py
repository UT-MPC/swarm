class DAGNode():
    def __init__(self, event_id) -> None:
        self.event_id = event_id
        self.isCompute = True # compute or communicate
        self.depends = []
    
    def depends_on(self, node) -> None:
        self.depends.append(node)