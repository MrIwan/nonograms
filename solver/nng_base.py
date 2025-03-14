import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import display, clear_output

class NNGBase:
    def __init__(self, size, row_restrictions, col_restrictions):
        self.size = size
        self.row_restrictions = row_restrictions
        self.col_restrictions = col_restrictions
        self.matrix = np.full((size, size), -1, dtype=np.int8)  # -1=unbekannt, 0=leer, 1=gefüllt
        self.history = []
        

    def solved(self) -> bool:
        """Ultra-optimierte Lösung mit:
        - Vectorized NumPy Operationen
        - Early Exit bei Fehlern
        - Minimalen Speicherzugriffen"""
        
        # Schnellcheck 1: Ungelöste Zellen
        if -1 in self.matrix:
            return False
        
        # Schnellcheck 2: Gesamtzahl der 1er
        if not self._check_total_sum():
            return False

        # Vectorized Block-Berechnung
        row_blocks = [self._get_blocks(row) for row in self.matrix]
        if any(r != a for r, a in zip(row_blocks, self.row_restrictions)):
            return False

        col_blocks = [self._get_blocks(col) for col in self.matrix.T]
        return all(c == a for c, a in zip(col_blocks, self.col_restrictions))

    def _check_total_sum(self) -> bool:
        """Vorab-Check der Gesamtsumme aller Blöcke"""
        total_row = sum(sum(b) for b in self.row_restrictions)
        total_col = sum(sum(b) for b in self.col_restrictions)
        if total_row != total_col:
            return False
        return np.count_nonzero(self.matrix == 1) == total_row

    @staticmethod
    def _get_blocks(arr: np.ndarray) -> list:
        """Vectorized Block-Erkennung mit NumPy"""
        padded = np.pad(arr, (1, 1), mode='constant')
        diffs = np.diff(padded)
        starts = np.where(diffs > 0)[0]
        ends = np.where(diffs < 0)[0]
        return (ends - starts).tolist()


    def show(self):
        """Konsolenausgabe des aktuellen Zustands"""
        symbols = {
            -1: '?',  # Unbekannt
            0: '□',   # Leer
            1: '■'    # Gefüllt
        }
        for row in self.matrix:
            print(' '.join(symbols[cell] for cell in row))
            
    def show_img(self, ax=None, **kwargs):
        """Visualisierung mit Matplotlib"""
        cmap = plt.cm.binary
        cmap.set_under('white')
        cmap.set_over('black')
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(5,5))
        
        ax.imshow(np.ma.masked_where(self.matrix == -1, self.matrix), 
                cmap=cmap, vmin=0, vmax=1, **kwargs)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(True, which='both', color='grey', linewidth=0.5)
        
    def save_history_frame(self):
        """Speichert den aktuellen Zustand in der History"""
        self.history.append(self.matrix.copy())

    def create_animation(self):
        """Erstellt eine Animation aus der History"""
        fig, ax = plt.subplots()
        frames = []

        def init():
            self.show_img(ax)
            return [ax.images[0]]

        def update(frame):
            ax.clear()
            self.matrix = frame
            self.show_img(ax)
            return [ax.images[0]]

        ani = animation.FuncAnimation(
            fig, update, frames=self.history,
            init_func=init, blit=True, interval=500
        )
        return ani

    def step(self) -> bool:
        """Muss in Unterklassen implementiert werden"""
        raise NotImplementedError

    def solve(self):
        """Löst das Nonogramm durch wiederholte step-Aufrufe"""
        self.save_history_frame()
        while not self.solved:
            self.step()
            self.save_history_frame()
        return self.matrix


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
