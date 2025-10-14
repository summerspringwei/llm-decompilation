

import angr
import capstone
import archinfo
import sys
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class AngrAnalyzer:
    def __init__(self, proj):
        self.proj = proj
        self.addr_to_inst_dict = {}


    def get_function_by_name(self, func_name):
        cfg = self.proj.analyses.CFGFast()
        target_function = None
        for _, func in cfg.functions.items():
            if func.name == func_name:
                target_function = func
                break
        if target_function:
            logger.debug(f"Found function: {target_function.name}")
        else:
            raise Exception(f"Function '{func_name}' not found")
        return target_function


    def get_all_address_in_function(self, func_name):
        target_function = self.get_function_by_name(func_name)
        function_cfg = target_function.graph
        for block_addr in function_cfg.nodes():
            block = self.proj.factory.block(block_addr.addr)
            for insn in block.capstone.insns:
                self.addr_to_inst_dict[insn.address] = insn


    def get_function_instrcution_address_range(self, func_name):
        target_function = self.get_function_by_name(func_name)
        first_instruction_addr = target_function.addr
        # Get the last instruction address of the function
        function_cfg = target_function.graph
        terminal_blocks = [node for node in function_cfg.nodes() if not list(function_cfg.successors(node))]
        
        ret_addr = -1
        for block_addr in terminal_blocks:
            block = self.proj.factory.block(block_addr.addr)
            for insn in block.capstone.insns:
                logger.debug(hex(insn.address), insn.mnemonic, insn.op_str)
                # Get target register for each instruction
                if insn.mnemonic == 'ret':
                    logger.debug(f"Return instruction found at {hex(insn.address)}")
                else:
                    # Get operands
                    for op in insn.operands:
                        if op.type == capstone.CS_OP_REG:  # Check if operand is a register
                            reg_name = insn.reg_name(op.reg)
                            logger.debug(f"Target register: {reg_name}")
            last_instruction_addr = block.capstone.insns[-1].address
            if last_instruction_addr > ret_addr:
                ret_addr = last_instruction_addr
        return (first_instruction_addr, ret_addr)


    def simulate(self, func_name, program_args, max_steps = 1000):
        # Create an initial state at program entry
        state = self.proj.factory.entry_state(args=program_args)
        # Create Simulation Manager
        simgr = self.proj.factory.simgr(state)

        # Get the function address range
        first_inst_addr, end_inst_addr = self.get_function_instrcution_address_range(func_name)
        
        step_count = 0
        while simgr.active and step_count < max_steps:
            state = simgr.active[0]
            addr = state.addr
            logger.debug(f"Step {step_count}: Current address: {hex(addr)}")
            if first_inst_addr <= addr < end_inst_addr:
                # Get and print current instruction
                reg_name = None
                inst = None
                if addr in self.addr_to_inst_dict.keys():
                    inst = self.addr_to_inst_dict[addr]
                    for op in inst.operands:
                        if op.type == capstone.CS_OP_REG:
                            reg_name = inst.reg_name(op.reg)
                if reg_name:
                    val = state.solver.eval(getattr(state.regs, reg_name))
                    print(f"Instruction at {hex(addr)}: {inst.mnemonic} {inst.op_str}, Target register: {reg_name}, value: {hex(val)}")
                    
            simgr.step(num_inst=1)
            step_count += 1
   
    def analyze_func_args(self, func_name):
        target_function = self.get_function_by_name(func_name)
        # Run variable recovery first
        vra = self.proj.analyses.VariableRecoveryFast(target_function)

        # Then run calling convention analysis with variable recovery results
        cca = self.proj.analyses.CallingConvention(target_function, analyze_callsites=False)

        # Print the results
        print(type(cca))
        print("Detected calling convention:", cca.cc)
        print("Detected number of arguments:", cca.cc.arch)
        for arg in cca.cc.int_args:
            print(f"Argument: {arg}")
    
    def analyze_func_by_define_use(self, func_name):
        # Run ReachingDefinitions analysis on the function
        target_function = self.get_function_by_name(func_name)
        rd = self.proj.analyses.ReachingDefinitions(target_function)
        
        # Print all uses for each instruction
        for node in target_function.nodes():
            block = self.proj.factory.block(node.addr)
            print(f"\nBlock at {hex(node.addr)}:")
            for insn in block.capstone.insns:
                print(f"\nInstruction: {insn.mnemonic} {insn.op_str}")
                # Get uses at this instruction address
                uses = rd.all_uses.get_uses_by_location(insn.address)
                if uses:
                    print("Uses:")
                    for use in uses:
                        print(f"  - {use}")
                else:
                    print("No uses found")
        # Print all definitions for each instruction
        # used_regs = set()

        # # Check each basic block in the function
        # for block_addr in target_function.block_addrs:
        #     # Get all ReachingDefinitions at the block's entry
        #     rd_at_block = rd.definitions_at.get(block_addr, [])
            
        #     # We'll also want to analyze the block's statements to see reads
        #     block = self.proj.factory.block(block_addr)
        #     irsb = block.vex
            
        #     # For each statement in the block
        #     for stmt in irsb.statements:
        #         # Look for reads of argument registers before any write to that register
        #         if stmt.tag == "Ist_WrTmp" and hasattr(stmt.data, "tag") and stmt.data.tag == "Iex_Get":
        #             # Get the register name accessed
        #             reg_offset = stmt.data.offset
        #             reg_name = self.proj.arch.register_names.get(reg_offset, None)
                    
        #             if reg_name in arg_regs:
        #                 # Check if this register was defined (written) earlier in this block or not
        #                 # If not overwritten before, it's a use of argument register
        #                 # Simplified: just record usage for now
        #                 used_regs.add(reg_name)

        # print(f"Argument registers used (likely function parameters): {used_regs}")
        # print(f"Number of parameters inferred: {len(used_regs)}")


def main():
    binary_name = 'example.out'
    # Load the binary
    binary_path = 'models/angr/example.out'
    func_name = 'foo'
    proj = angr.Project(binary_path, auto_load_libs=False)
    # Analyze the binary to build a Control Flow Graph (CFG)
    analyzer = AngrAnalyzer(proj)
    analyzer.get_all_address_in_function(func_name)
    analyzer.simulate(func_name, program_args=[binary_name, '17', '29'])
    analyzer.analyze_func_args(func_name)
    analyzer.analyze_func_by_define_use(func_name)

if __name__ == "__main__":
    main()
