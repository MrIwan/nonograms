import numpy as np
from tabulate import tabulate
import time
from .create_nng import create_restrictions_from_array

class NNGBase:
    def __init__(self, row_restrictions, col_restrictions, save_history = False):
        self.rows = len(row_restrictions)
        self.colums = len(col_restrictions)
        self.row_restrictions = row_restrictions
        self.col_restrictions = col_restrictions
        self.matrix = np.full((self.rows, self.colums), -1, dtype=np.int8)  # -1=unbekannt, 0=leer, 1=gefüllt
        self.save_history = save_history
        self.history = []
        self.start_time = 0
        self.cpu_time = 0

    def solved(self, temp_matrix = None) -> bool:

        if temp_matrix.all() == None:
            temp_matrix = self.matrix
        temp_row_restrictions, temp_col_restrictions = create_restrictions_from_array(temp_matrix)
        if temp_row_restrictions == self.row_restrictions and temp_col_restrictions == self.col_restrictions:
            return True
        return False


    def show(self):

        """Konsolenausgabe des aktuellen Zustands"""
        symbols = {
            -1: '?',  # Unbekannt
            0: '□',   # Leer
            1: '■'    # Gefüllt
        }

        max_len_col_res = max(len(a) for a in self.col_restrictions)
        col_padded = np.array([[' '] * (max_len_col_res - len(subarray)) + subarray for subarray in self.col_restrictions]).T

        max_len_row_res = max([len(a) for a in self.row_restrictions])
        row_padded = np.array([[' '] * (max_len_row_res - len(subarray)) + subarray  for subarray in self.row_restrictions])

        padding = np.full((max_len_col_res, max_len_row_res), ' ')
        matrix_translated = np.array([[symbols[i] for i in row] for row in self.matrix])

        full_matrix = np.hstack(
            (np.vstack((padding, row_padded)),
            np.vstack((col_padded, matrix_translated))))

        print(tabulate(full_matrix))
        print(f'cpu_time = {self.cpu_time}')

    def add_history_frame(self, add_matrix: np.ndarray = None):
        if self.save_history:
            if add_matrix.any():
                self.history.append(add_matrix.copy())
            else:
                self.history.append(self.matrix.copy())

    def start_timer(self):
        self.start_time = time.process_time()

    def stop_timer(self):
        self.cpu_time = time.process_time() - self.start_time

    def solve(self):
        """Muss in den Unterklassen implementiert werden"""
        raise NotImplementedError
    

class NNGManual(NNGBase):
    def __init__(self, size, row_restrictions, col_restrictions, save_history=False):
        super().__init__(size, row_restrictions, col_restrictions, save_history)

    def _step(self) -> bool:
        self.show()
        eingabe = input('Bitte gib die Coordinaten von dem Wert ein und den Wert ( -1 = unbekannt, 0 = leer, 1 = makiert)').split(' ')
        if eingabe[0] == 'q':
            return True
        else:
            try:
                self.matrix[int(eingabe[0])][int(eingabe[1])] = int(eingabe[2])
            except:
                print('Falsche eingabe!')
        return
    
    def solve(self):
        solved = False
        """Muss in den Unterklassen implementiert werden"""
        while not solved:
            self._step()
            self.add_history_frame()
            solved = self.solved()
        print("Lösung")
        self.show()
        return None
    
class NNGStupidRec(NNGBase):
    def __init__(self, size, row_restrictions, col_restrictions, save_history=False):
        super().__init__(size, row_restrictions, col_restrictions, save_history)

    def _rec_step(self, temp_matrix: np.ndarray, i, j):
        self.add_history_frame(temp_matrix)
        if temp_matrix.min() != -1 and self.solved(temp_matrix):
            self.matrix = temp_matrix.copy()
            print('Lösung gefunden')
            return True
        for x in [0, 1]:
            next_i = i + 1
            next_j = j
            if next_i >= self.size:
                next_i = 0
                next_j = j + 1
            if next_j < self.size:
                new_matrix = temp_matrix.copy()
                new_matrix[next_i][next_j] = x
                if self._rec_step(new_matrix, next_i, next_j):
                    return True
        return False
    
    def solve(self):
        """Pretty Stupid Recursive Algorithm"""
        self.start_timer()
        self._rec_step(self.matrix.copy(), -1, 0)
        self.stop_timer()
        self.show()
        


class NNGreedy(NNGBase):
    """Greedy-Algorithmus Implementierung"""
    def __init__(self, size, row_restrictions, col_restrictions, save_history=False):
        super().__init__(size, row_restrictions, col_restrictions, save_history)
        
    def solve(self) -> bool:
        return NotImplementedError
    

class NNGSat(NNGBase):
    """SAT-Implementierung"""
    def __init__(self, size, row_restrictions, col_restrictions, save_history=False):
        super().__init__(size, row_restrictions, col_restrictions, save_history)
        
    def solve(self) -> bool:
        return NotImplementedError

