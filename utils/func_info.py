

class FuncInfo:
    def __init__(self, record: dict):
        if (len(record["func_info"]["functions"]) == 0):
            print(record["func_info"])
        assert len(record["func_info"]["functions"]) == 1, "Only one function is allowed in the record"
        func_info = record["func_info"]["functions"][0]
        self.func_name: str = func_info["name"]
        self.called_functions: list[str] = func_info["called_functions"]
        self.num_loops: int = func_info["num_loops"]
        self.has_defined_structs: bool = func_info["has_defined_structs"]
        self.has_globals: bool = func_info["has_globals"]
        self.struct_args: bool = func_info["struct_args"]
        self.unused_args: list[int] = func_info["unused_args"]
        self.bb_count_list = record["llvm_ir"]["bb_count"]["bb_list_size"]
        

    def __eq__(self, other):
        return self.called_functions == other.called_functions and self.num_loops == other.num_loops and self.num_branches == other.num_branches and self.num_instructions == other.num_instructions

    def __hash__(self):
        return hash((self.called_functions, self.num_loops, self.num_branches, self.num_instructions))