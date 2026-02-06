# ============================================
# DIJKSTRA AVEC NETWORKX + VISUALISATION
# ============================================

import networkx as nx
import math
import matplotlib.pyplot as plt

# ============================================
# ÉTAPE 1 : CRÉATION DU GRAPHE
# ============================================

G = nx.Graph()

# Ajout des arêtes avec poids
edges = [
    (0, 3, 2.1),
    (0, 2, 2.0),
    (0, 1, 2.5),
    (3, 5, 2.5),
    (1, 4, 1.0),
    (2, 4, 0.6),
    (2, 5, 1.5),
    (5, 7, 2.0),
    (5, 6, 1.9),
    (6, 8, 1.7),
    (7, 8, 2.0),
    (4, 6, 2.3),
]

for a, b, w in edges:
    G.add_edge(a, b, weight=w)

print("✓ Graphe créé")

# ============================================
# ÉTAPE 2 : DIJKSTRA (IMPLÉMENTATION MAISON)
# ============================================

def dijkstra(G, depart):
    distances = {node: math.inf for node in G.nodes}
    parents = {node: None for node in G.nodes}
    visites = set()

    distances[depart] = 0

    while len(visites) < len(G.nodes):
        noeud_min = None
        distance_min = math.inf

        for node in G.nodes:
            if node not in visites and distances[node] < distance_min:
                distance_min = distances[node]
                noeud_min = node

        if noeud_min is None:
            break

        visites.add(noeud_min)

        for voisin, data in G[noeud_min].items():
            poids = data["weight"]
            nouvelle_distance = distances[noeud_min] + poids

            if nouvelle_distance < distances[voisin]:
                distances[voisin] = nouvelle_distance
                parents[voisin] = noeud_min

    return distances, parents

# ============================================
# ÉTAPE 3 : RECONSTRUCTION DU CHEMIN
# ============================================

def reconstruire_chemin(parents, depart, arrivee):
    chemin = []
    courant = arrivee

    while courant is not None:
        chemin.append(courant)
        courant = parents[courant]

    chemin.reverse()

    if chemin[0] != depart:
        return None

    return chemin

# ============================================
# ÉTAPE 4 : EXÉCUTION
# ============================================

depart = 0
arrivee = 8

distances, parents = dijkstra(G, depart)
chemin = reconstruire_chemin(parents, depart, arrivee)

print("\n============================================")
print("PLUS COURT CHEMIN")
print("============================================")

print("Chemin :", " → ".join(map(str, chemin)))
print(f"Distance totale : {distances[arrivee]:.2f}")

# ============================================
# ÉTAPE 5 : DESSIN DU GRAPHE + CHEMIN
# ============================================

plt.figure(figsize=(10, 7))

# Positionnement automatique (lisible)
pos = nx.spring_layout(G, seed=42)

# Dessin du graphe
nx.draw(
    G,
    pos,
    with_labels=True,
    node_size=1200,
    font_size=12,
    font_weight="bold"
)

# Affichage des poids
labels = nx.get_edge_attributes(G, "weight")
nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)

# Mise en évidence du chemin le plus court
chemin_edges = list(zip(chemin, chemin[1:]))

nx.draw_networkx_edges(
    G,
    pos,
    edgelist=chemin_edges,
    width=4
)

plt.title("Graphe des chemins – Plus court chemin mis en évidence")
plt.show()

print("\n✓ Programme terminé")
