import numpy as np
from tabulate import tabulate
import time

class NNGBase:
    def __init__(self, size, row_restrictions, col_restrictions, save_history = False):
        self.size = size
        self.row_restrictions = row_restrictions
        self.col_restrictions = col_restrictions
        self.matrix = np.full((size, size), -1, dtype=np.int8)  # -1=unbekannt, 0=leer, 1=gefüllt
        self.save_history = save_history
        self.history = []
        self.start_time = 0
        self.cpu_time = 0

    def solved(self, temp_matrix = None) -> bool:

        if temp_matrix.all() == None:
            temp_matrix = self.matrix

        temp_row_restrictions = []
        temp_col_restrictions = []
        for i in range(self.size):
            temp_row_restrictions.append([])
            temp_col_restrictions.append([])
            for j in range(self.size):
                if temp_matrix[i][j] == 1:
                    if len(temp_row_restrictions[i]) == 0:
                        temp_row_restrictions[i].append(1)
                    elif temp_matrix[i][j-1] == 1:
                        temp_row_restrictions[i][len(temp_row_restrictions[i]) - 1] += 1
                    else:
                        temp_row_restrictions[i].append(1)

                if temp_matrix[j][i] == 1:
                    if len(temp_col_restrictions[i]) == 0:
                        temp_col_restrictions[i].append(1)
                    elif temp_matrix[j-1][i] == 1:
                        temp_col_restrictions[i][len(temp_col_restrictions[i]) - 1] += 1
                    else:
                        temp_col_restrictions[i].append(1)

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
    def __init__(self, size):
        super().__init__(size)
        
    def step(self) -> bool:
        # Implementiere hier deine Greedy-Logik
        # Beispiel: Fülle offensichtliche Zellen basierend auf Beschränkungen
        old_matrix = self.matrix.copy()
        
        # Platzhalter-Logik
        changed = False
        for i in range(self.size):
            # Hier würde die eigentliche Greedy-Logik stehen
            if np.all(self.matrix[i] == -1):
                self.matrix[i] = 0
                changed = True
                
        self.solved = np.all(self.matrix != -1)
        return not np.array_equal(old_matrix, self.matrix)


class NNGBacktracking(NNGBase):
    """Backtracking Implementierung"""
    def __init__(self, size):
        super().__init__(size)
        
    def step(self) -> bool:
        # Implementiere Backtracking-Logik
        # (Für Backtracking wäre eine rekursive Implementierung typischer,
        # aber hier als iterative Version)
        pass
