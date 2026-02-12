import heapq


class Maze:
    def __init__(self, filename):
        """
        Charge le labyrinthe depuis un fichier texte.
        - filename: chemin vers le fichier du labyrinthe
        
        Symboles:
        - '#' = mur (non traversable)
        - '.' = case libre
        - 'A' = départ
        - 'B' = arrivée
        """
        self.grid = []  # Le labyrinthe sous forme de liste 2D
        self.start = None  # Coordonnées du point de départ (i, j)
        self.goal = None   # Coordonnées du point d'arrivée (i, j)
        
        with open(filename, 'r') as f:
            for line in f:
                # On enlève le '\n' à la fin et on convertit en liste de caractères
                self.grid.append(list(line.strip()))
        
        self.height = len(self.grid)      # Nombre de lignes
        self.width = len(self.grid[0])    # Nombre de colonnes
        
        # Matrice de récompense (coût pour traverser chaque case)
        # Par défaut: 1 pour case libre, infini pour mur
        self.rewards = [[0 for _ in range(self.width)] for _ in range(self.height)]
        
        # Parcourir la grille pour trouver A, B et initialiser les récompenses
        for i in range(self.height):
            for j in range(self.width):
                cell = self.grid[i][j]
                if cell == 'A':
                    self.start = (i, j)
                    self.rewards[i][j] = 1  # Coût de traversée
                elif cell == 'B':
                    self.goal = (i, j)
                    self.rewards[i][j] = 1  # Coût de traversée
                elif cell == '#':
                    self.rewards[i][j] = float('inf')  # Mur = coût infini
                else:
                    self.rewards[i][j] = 1  # Case libre = coût 1
    
    def in_bounds(self, i, j):
        """
        Vérifie si la cellule (i, j) appartient à la grille.
        - i: numéro de ligne
        - j: numéro de colonne
        Retourne True si la cellule est dans les limites de la grille.
        """
        return 0 <= i < self.height and 0 <= j < self.width
    
    def is_valid(self, i, j):
        """
        Vérifie si la position (i, j) est valide et traversable.
        - i: numéro de ligne
        - j: numéro de colonne
        Retourne True si on peut marcher sur cette case.
        """
        # Vérifier que c'est dans les limites
        if not self.in_bounds(i, j):
            return False
        # Vérifier que ce n'est pas un mur
        return self.grid[i][j] != '#'
    
    def get_neighbors(self, i, j):
        """
        Retourne la liste des voisins valides de la case (i, j).
        On peut se déplacer en haut, bas, gauche, droite.
        """
        neighbors = []
        # Les 4 directions possibles: (delta_i, delta_j)
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # haut, bas, gauche, droite
        
        for di, dj in directions:
            ni, nj = i + di, j + dj  # nouvelle position
            if self.is_valid(ni, nj):
                neighbors.append((ni, nj))
        
        return neighbors
    
    def solve(self, si, sj, gi, gj):
        """
        Trouve le plus court chemin de (si, sj) à (gi, gj) avec Dijkstra.
        
        Paramètres:
        - si, sj: position de départ (ligne, colonne)
        - gi, gj: position d'arrivée (ligne, colonne)
        
        Retourne:
        - Le chemin sous forme de liste de positions [(i1,j1), (i2,j2), ...]
        - None si aucun chemin n'existe
        """
        # Distance minimale pour atteindre chaque case
        dist = {}
        dist[(si, sj)] = 0
        
        # Pour reconstruire le chemin: parent[case] = case précédente
        parent = {}
        parent[(si, sj)] = None
        
        # File de priorité: (distance, (i, j))
        heap = [(0, (si, sj))]
        
        # Cases déjà visitées
        visited = set()
        
        while heap:
            # Prendre la case avec la plus petite distance
            current_dist, (i, j) = heapq.heappop(heap)
            
            # Si déjà visitée, on passe
            if (i, j) in visited:
                continue
            
            # Marquer comme visitée
            visited.add((i, j))
            
            # Si on est arrivé au but, reconstruire le chemin
            if (i, j) == (gi, gj):
                return self._reconstruct_path(parent, (gi, gj))
            
            # Explorer les voisins
            for (ni, nj) in self.get_neighbors(i, j):
                if (ni, nj) not in visited:
                    # Coût = 1 pour chaque déplacement
                    new_dist = current_dist + 1
                    
                    # Si on trouve un meilleur chemin
                    if (ni, nj) not in dist or new_dist < dist[(ni, nj)]:
                        dist[(ni, nj)] = new_dist
                        parent[(ni, nj)] = (i, j)
                        heapq.heappush(heap, (new_dist, (ni, nj)))
        
        # Aucun chemin trouvé
        return None
    
    def _reconstruct_path(self, parent, goal):
        """
        Reconstruit le chemin en remontant les parents.
        """
        path = []
        current = goal
        while current is not None:
            path.append(current)
            current = parent[current]
        path.reverse()  # On inverse pour avoir départ -> arrivée
        return path
    
    def display(self, path=None):
        """
        Affiche le labyrinthe. Si un chemin est donné, le marque avec '*'.
        """
        # Copie du grid pour ne pas le modifier
        display_grid = [row[:] for row in self.grid]
        
        if path:
            for (i, j) in path:
                if display_grid[i][j] not in ('A', 'B'):
                    display_grid[i][j] = '*'
        
        for row in display_grid:
            print(''.join(row))


# Code de test
if __name__ == "__main__":
    # Charger le labyrinthe
    maze = Maze("labyrinthe.txt")
    
    print("Labyrinthe original:")
    maze.display()
    
    print(f"\nDépart: {maze.start}, Arrivée: {maze.goal}")
    print(f"Dimensions: {maze.height} x {maze.width}")
    
    # Résoudre le labyrinthe
    path = maze.solve(maze.start[0], maze.start[1], maze.goal[0], maze.goal[1])
    
    if path:
        print(f"\nChemin trouvé ({len(path)} cases):")
        maze.display(path)
    else:
        print("\nAucun chemin trouvé!")
