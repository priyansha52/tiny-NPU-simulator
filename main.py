from enum import Enum, auto
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

class OpCode(Enum):
    LOAD = auto()
    STORE = auto()
    MAC = auto()
    ADD = auto()
    MUL = auto()
    NOP = auto()

class Instruction:
    def __init__(self, opcode, operands=None):
        self.opcode = opcode
        self.operands = operands or []

    def __repr__(self):
        return f"{self.opcode.name} {self.operands}"

class Memory:
    def __init__(self, size=256):
        self.data = [0] * size

    def load(self, addr):
        return self.data[addr]

    def store(self, addr, value):
        self.data[addr] = value

    def __repr__(self):
        return f"Memory{self.data[:10]}..."

class PE:
    def __init__(self, pe_id):
        self.id = pe_id
        self.registers = {"reg0": 0, "reg1": 0, "reg2": 0}

    def execute(self, instruction, memory=None):
        op = instruction.opcode
        args = instruction.operands

        if op == OpCode.LOAD:
            reg, addr = args
            self.registers[reg] = memory.load(addr)
        elif op == OpCode.STORE:
            reg, addr = args
            memory.store(addr, self.registers[reg])
        elif op == OpCode.MAC:
            r_dest, r_a, r_b = args
            self.registers[r_dest] += self.registers[r_a] * self.registers[r_b]
        elif op == OpCode.ADD:
            r_dest, r_a, r_b = args
            self.registers[r_dest] = self.registers[r_a] + self.registers[r_b]
        elif op == OpCode.MUL:
            r_dest, r_a, r_b = args
            self.registers[r_dest] = self.registers[r_a] * self.registers[r_b]
        elif op == OpCode.NOP:
            pass

    def __repr__(self):
        return f"PE{self.id}{self.registers}"

class SystolicArray:
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        self.grid = [[PE(pe_id=f"{r}-{c}") for c in range(cols)] for r in range(rows)]

    def get_pe(self, r, c):
        return self.grid[r][c]

    def __repr__(self):
        lines = []
        for row in self.grid:
            lines.append(" | ".join(str(pe.registers["reg2"]) for pe in row))
        return "\n".join(lines)

class CycleCounter:
    def __init__(self):
        self.cycles = 0

    def tick(self):
        self.cycles += 1

    def __repr__(self):
        return f"Total cycles: {self.cycles}"

def naive_matmul_cycles(A, B):
    n = len(A)
    clock = CycleCounter()
    pe = PE(pe_id="single")

    for r in range(n):
        for c in range(n):
            pe.registers["reg2"] = 0
            for k in range(n):
                pe.registers["reg0"] = A[r][k]
                pe.registers["reg1"] = B[k][c]
                pe.execute(Instruction(OpCode.MAC, ["reg2", "reg0", "reg1"]))
                clock.tick()

    return clock.cycles

def animate_array(snapshots):
    n = len(snapshots[0])
    fig, ax = plt.subplots()
    vmax = max(max(row) for row in snapshots[-1])

    def update(frame):
        ax.clear()
        values = snapshots[frame]
        im = ax.imshow(values, cmap="viridis", vmin=0, vmax=vmax)
        for r in range(n):
            for c in range(n):
                ax.text(c, r, values[r][c], ha="center", va="center", color="white", fontsize=14)
        ax.set_title(f"Cycle {frame + 1} / {len(snapshots)}")
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        return [im]

    ani = animation.FuncAnimation(fig, update, frames=len(snapshots), interval=800, repeat=True)
    ani.save("pe_animation.gif", writer="pillow")
    plt.show()

if __name__ == "__main__":
    A = [[1, 2],
         [3, 4]]
    B = [[5, 6],
         [7, 8]]

    n = len(A)
    array = SystolicArray(rows=n, cols=n)
    mem = Memory()
    clock = CycleCounter()

    for r in range(n):
        for c in range(n):
            array.get_pe(r, c).registers["reg2"] = 0

    snapshots = []

    for k in range(n):
        for r in range(n):
            for c in range(n):
                pe = array.get_pe(r, c)
                pe.registers["reg0"] = A[r][k]
                pe.registers["reg1"] = B[k][c]
                pe.execute(Instruction(OpCode.MAC, ["reg2", "reg0", "reg1"]))
        clock.tick()
        snapshot = [[array.get_pe(r, c).registers["reg2"] for c in range(n)] for r in range(n)]
        snapshots.append(snapshot)

    print("Simulated result:")
    print(array)
    print(clock)

    expected = np.matmul(A, B)
    print("\nExpected (numpy):")
    print(expected)
    naive_cycles = naive_matmul_cycles(A, B)
    print(f"\nNaive single-core cycles: {naive_cycles}")
    print(f"Systolic array cycles: {clock.cycles}")
    print(f"Speedup: {naive_cycles / clock.cycles:.2f}x")

    animate_array(snapshots)