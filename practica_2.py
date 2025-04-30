import requests
from bs4 import BeautifulSoup
import networkx as nx
import matplotlib.pyplot as plt

# Configuración
url_principal = "https://news.ycombinator.com/"
max_links_principales = 10
max_subenlaces = 3
timeout_segundos = 5
headers = {'User-Agent': 'Mozilla/5.0'}

# Función para extraer enlaces desde una URL
def get_links(url, limit):
    try:
        response = requests.get(url, headers=headers, timeout=timeout_segundos)
        soup = BeautifulSoup(response.text, 'html.parser')
        links = [a['href'] for a in soup.find_all('a', href=True) if a['href'].startswith('http')]
        return links[:limit]
    except Exception as e:
        print(f"[ERROR] No se pudo acceder a {url}: {e}")
        return []

# Inicialización
visitados = set()
G = nx.DiGraph()

# Extraer enlaces del sitio principal
enlaces = get_links(url_principal, max_links_principales)

# Construir grafo de enlaces
for enlace in enlaces:
    if enlace in visitados:
        continue
    visitados.add(enlace)
    G.add_edge(url_principal, enlace)

    sub_enlaces = get_links(enlace, max_subenlaces)
    for sub in sub_enlaces:
        if sub not in visitados:
            G.add_edge(enlace, sub)
            visitados.add(sub)

# Mostrar enlaces entrantes/salientes
print("\n📊 Conteo de enlaces por página:")
for nodo in G.nodes():
    print(f"- {nodo}")
    print(f"  ↳ Entrantes: {G.in_degree(nodo)}")
    print(f"  ↳ Salientes: {G.out_degree(nodo)}")

# Calcular PageRank
pagerank = nx.pagerank(G)

# Mostrar Top 5
top_5 = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:5]
print("\n⭐ Top 5 páginas más importantes según PageRank:")
for i, (url, score) in enumerate(top_5, start=1):
    print(f"{i}. {url} → {score:.5f}")

# Visualización del grafo
plt.figure(figsize=(12, 10))
pos = nx.spring_layout(G, k=0.5)
nx.draw(G, pos, with_labels=False, node_size=300, arrowsize=10, alpha=0.7)
nx.draw_networkx_labels(G, pos, font_size=7)
plt.title("🌐 Grafo de enlaces web (Crawler + PageRank)", fontsize=14)
plt.show()
