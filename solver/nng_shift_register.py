from .nng_base import NNGBase
from dataclasses import dataclass
import numpy as np

@dataclass
class Block:
    row: int
    number: int
    length: int
    start_pos: int

    def end_pos(self) -> int:
        return self.start_pos + self.length

class NNGShitRegister(NNGBase):
    def __init__(self, row_restrictions, col_restrictions, save_history=False):
        super().__init__(row_restrictions, col_restrictions, save_history)
        self.blocks: list[Block] = self._create_blocks(row_restrictions)
        self.active_block = len(self.blocks) - 1
        self.tryed = []

    def _create_blocks(self, row_restrictions) -> list[Block]:
        blocks: list[Block] = []
        # create Array with all Blocks
        for i, rr in enumerate(row_restrictions):
            current_shift = 0
            for j, n in enumerate(rr):
                blocks.append(Block(i, j, n, current_shift))
                current_shift += n + 1
        return blocks
    
    def _create_matrix(self):
        temp_matrix = np.zeros((len(self.row_restrictions), len(self.col_restrictions)), dtype=int)
        for b in self.blocks:
            for i in range(b.length):
                temp_matrix[b.row][b.start_pos + i] = 1
        return temp_matrix
    
    def _shift_is_possible(self) -> bool:
        # check if block would be pushed out of the board
        if self.blocks[self.active_block].end_pos() >= len(self.row_restrictions):
            return False
        # check if next Block ist blocking the current Block
        if self.active_block < len(self.blocks) - 1:
            if self.blocks[self.active_block].row == self.blocks[self.active_block + 1].row:
                if self.blocks[self.active_block].end_pos() + 1 >= self.blocks[self.active_block + 1].start_pos:
                    return False
        return True

    def _shift(self):
        if self._shift_is_possible():
            self.blocks[self.active_block].start_pos += 1
        else:
            while not self._shift_is_possible():
                self.active_block -= 1
                if self.active_block == -1:
                    print('NNG nicht lösbar')
            self.blocks[self.active_block].start_pos += 1

            # reset all following blocks
            current_row = self.blocks[self.active_block].row
            current_shift = self.blocks[self.active_block].end_pos() + 1
            for i in range(self.active_block + 1, len(self.blocks)):
                if current_row != self.blocks[i].row:
                    current_shift = 0
                    current_row = self.blocks[i].row
                
                self.blocks[i].start_pos = current_shift
                current_shift = self.blocks[i].end_pos() + 1
            self.active_block = len(self.blocks) - 1
    
    def solve(self):
        self.start_timer()
        solved = False
        while not solved:
            if self.solved(self._create_matrix()):
                solved = True
                self.matrix = self._create_matrix()
                self.stop_timer()
                self.show()
            else:
                self._shift()