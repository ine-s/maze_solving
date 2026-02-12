import heapq
import random


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
                stripped = line.strip()
                if stripped:  # Ignorer les lignes vides
                    self.grid.append(list(stripped))
        
        self.height = len(self.grid)      # Nombre de lignes
        self.width = len(self.grid[0]) if self.grid else 0    # Nombre de colonnes
        
        # Matrice de récompense (coût pour traverser chaque case)
        # Par défaut: 1 pour case libre, infini pour mur
        self.rewards = [[0 for _ in range(self.width)] for _ in range(self.height)]
        
        # Parcourir la grille pour trouver A, B et initialiser les récompenses
        for i in range(self.height):
            for j in range(len(self.grid[i])):  # Utiliser la longueur réelle de chaque ligne
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
    
    # ==================== PARTIE B : Fonctions de génération ====================
    
    @classmethod
    def create_empty(cls, height, width, start=(1, 1), goal=None):
        """
        Crée un labyrinthe vide (sans charger de fichier).
        
        Paramètres:
        - height: nombre de lignes
        - width: nombre de colonnes
        - start: tuple (i, j) pour le départ
        - goal: tuple (i, j) pour l'arrivée (par défaut: coin opposé)
        
        Retourne: une instance de Maze
        """
        maze = cls.__new__(cls)
        maze.height = height
        maze.width = width
        maze.start = start
        maze.goal = goal if goal else (height - 2, width - 2)
        
        # Créer une grille vide avec murs sur les bords
        maze.grid = []
        for i in range(height):
            row = []
            for j in range(width):
                if i == 0 or i == height - 1 or j == 0 or j == width - 1:
                    row.append('#')  # Mur sur les bords
                else:
                    row.append('.')  # Case libre
            maze.grid.append(row)
        
        # Placer départ et arrivée
        maze.grid[maze.start[0]][maze.start[1]] = 'A'
        maze.grid[maze.goal[0]][maze.goal[1]] = 'B'
        
        # Initialiser la matrice de récompense
        maze.rewards = [[0 for _ in range(width)] for _ in range(height)]
        maze.init_rewards()
        
        return maze
    
    def generate_obstacles_random(self, density=0.2, seed=None):
        """
        Génère des obstacles de manière aléatoire.
        
        Paramètres:
        - density: proportion de cases qui deviennent des murs (0.0 à 1.0)
        - seed: graine pour reproductibilité (optionnel)
        
        Les cellules de départ et d'arrivée restent toujours franchissables.
        """
        if seed is not None:
            random.seed(seed)
        
        for i in range(1, self.height - 1):
            for j in range(1, self.width - 1):
                # Ne pas toucher au départ et à l'arrivée
                if (i, j) == self.start or (i, j) == self.goal:
                    continue
                
                # Placer un mur avec probabilité = density
                if random.random() < density:
                    self.grid[i][j] = '#'
                else:
                    self.grid[i][j] = '.'
        
        # Mettre à jour les récompenses
        self.init_rewards()
    
    def generate_obstacles_deterministic(self, pattern='grid', spacing=4):
        """
        Génère des obstacles de manière déterministe selon un motif.
        
        Paramètres:
        - pattern: 'grid' (grille), 'horizontal' (lignes), 'vertical' (colonnes)
        - spacing: espacement entre les obstacles
        
        Les cellules de départ et d'arrivée restent toujours franchissables.
        """
        # D'abord, nettoyer la grille (garder seulement les bords)
        for i in range(1, self.height - 1):
            for j in range(1, self.width - 1):
                self.grid[i][j] = '.'
        
        for i in range(1, self.height - 1):
            for j in range(1, self.width - 1):
                # Ne pas toucher au départ et à l'arrivée
                if (i, j) == self.start or (i, j) == self.goal:
                    continue
                
                place_wall = False
                
                if pattern == 'grid':
                    # Grille: murs aux intersections régulières
                    if i % spacing == 0 and j % spacing == 0:
                        place_wall = True
                elif pattern == 'horizontal':
                    # Lignes horizontales avec passages
                    if i % spacing == 0 and j % (spacing * 2) != 0:
                        place_wall = True
                elif pattern == 'vertical':
                    # Colonnes verticales avec passages
                    if j % spacing == 0 and i % (spacing * 2) != 0:
                        place_wall = True
                
                if place_wall:
                    self.grid[i][j] = '#'
        
        # Replacer départ et arrivée
        self.grid[self.start[0]][self.start[1]] = 'A'
        self.grid[self.goal[0]][self.goal[1]] = 'B'
        
        # Mettre à jour les récompenses
        self.init_rewards()
    
    def init_rewards(self, move_penalty=-1, goal_reward=100, bonus_cells=None):
        """
        Initialise la matrice de récompense.
        
        Paramètres:
        - move_penalty: pénalité pour chaque déplacement (valeur négative)
        - goal_reward: récompense pour atteindre l'arrivée
        - bonus_cells: dict {(i,j): bonus} pour des cellules avec bonus
        
        La pénalité encourage les chemins courts.
        """
        if bonus_cells is None:
            bonus_cells = {}
        
        for i in range(self.height):
            for j in range(len(self.grid[i])):
                cell = self.grid[i][j]
                
                if cell == '#':
                    # Mur = coût infini (inaccessible)
                    self.rewards[i][j] = float('-inf')
                elif (i, j) == self.goal:
                    # Arrivée = grande récompense
                    self.rewards[i][j] = goal_reward
                elif (i, j) in bonus_cells:
                    # Cellule avec bonus
                    self.rewards[i][j] = bonus_cells[(i, j)]
                else:
                    # Case normale = pénalité de déplacement
                    self.rewards[i][j] = move_penalty
    
    def add_bonus(self, i, j, bonus_value):
        """
        Ajoute un bonus sur une cellule spécifique.
        
        Paramètres:
        - i, j: coordonnées de la cellule
        - bonus_value: valeur du bonus (positif)
        """
        if self.in_bounds(i, j) and self.grid[i][j] not in ('#', 'A', 'B'):
            self.rewards[i][j] = bonus_value
    
    def save_to_file(self, filename):
        """
        Sauvegarde le labyrinthe dans un fichier texte.
        """
        with open(filename, 'w') as f:
            for row in self.grid:
                f.write(''.join(row) + '\n')
    
    # ==================== FIN PARTIE B ====================
    
    # ==================== PARTIE C : Algorithme Dijkstra (à implémenter plus tard) ====================
    # def solve(self, si, sj, gi, gj):
    #     """À implémenter dans la Partie C"""
    #     pass
    
    # def _reconstruct_path(self, parent, goal):
    #     """À implémenter dans la Partie C"""
    #     pass
    # ==================== FIN PARTIE C ====================
    
    def display(self, path=None):
        """
        Affiche le labyrinthe avec des couleurs.
        - Chemin: vert
        - Départ (A): bleu
        - Arrivée (B): rouge
        - Murs (#): gris
        - Cases libres (.): blanc
        """
        # Codes couleur ANSI
        RESET = '\033[0m'
        GREEN = '\033[92m'   # Chemin
        BLUE = '\033[94m'    # Départ
        RED = '\033[91m'     # Arrivée
        GRAY = '\033[90m'    # Murs
        YELLOW = '\033[93m'  # Symbole du chemin
        
        # Convertir le chemin en set pour recherche rapide
        path_set = set(path) if path else set()
        
        for i in range(self.height):
            line = ""
            for j in range(len(self.grid[i])):  # Utiliser la longueur réelle de chaque ligne
                cell = self.grid[i][j]
                
                if (i, j) in path_set:
                    if cell == 'A':
                        line += BLUE + 'A' + RESET
                    elif cell == 'B':
                        line += RED + 'B' + RESET
                    else:
                        line += GREEN + '●' + RESET  # Symbole du chemin
                elif cell == '#':
                    line += GRAY + '█' + RESET
                elif cell == 'A':
                    line += BLUE + 'A' + RESET
                elif cell == 'B':
                    line += RED + 'B' + RESET
                else:
                    line += ' '  # Case libre = espace
            print(line)


# Code de test
if __name__ == "__main__":
    print("=" * 50)
    print("TEST PARTIE A : Chargement depuis fichier")
    print("=" * 50)
    
    # Charger le labyrinthe
    maze = Maze("labyrinthe.txt")
    
    print("Labyrinthe original:")
    maze.display()
    
    print(f"\nDépart: {maze.start}, Arrivée: {maze.goal}")
    print(f"Dimensions: {maze.height} x {maze.width}")
    
    # L'algorithme de résolution sera implémenté dans la Partie C
    
    print("\n" + "=" * 50)
    print("TEST PARTIE B : Génération de labyrinthe")
    print("=" * 50)
    
    # Créer un labyrinthe vide 25x25
    maze2 = Maze.create_empty(25, 25, start=(1, 1), goal=(23, 23))
    
    print("\n--- Labyrinthe avec obstacles ALÉATOIRES (densité 15%) ---")
    maze2.generate_obstacles_random(density=0.15, seed=42)
    maze2.display()
    
    # Test génération déterministe
    print("\n--- Labyrinthe avec obstacles DÉTERMINISTES (grille) ---")
    maze3 = Maze.create_empty(25, 25, start=(1, 1), goal=(23, 23))
    maze3.generate_obstacles_deterministic(pattern='grid', spacing=5)
    maze3.display()
    
    # Afficher la matrice de récompense
    print("\n--- Matrice de récompense (extrait 5x5) ---")
    print("Légende: -1 = pénalité déplacement, 100 = arrivée, -inf = mur")
    for i in range(5):
        row = [f"{maze3.rewards[i][j]:>5}" if maze3.rewards[i][j] != float('-inf') else " -inf" 
               for j in range(5)]
        print(" ".join(row))
