import pandas as pd
import requests
import random
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Paso 1: Extraer enlaces desde Hacker News
def get_links(url, limit=20):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    links = [a['href'] for a in soup.find_all('a', href=True) if a['href'].startswith('http')]
    return links[:limit]

# Simular logs con navegación de múltiples IPs
def simular_logs(num_ips=30):
    links = get_links("https://news.ycombinator.com/")
    registros = []
    for i in range(num_ips):
        ip = f"192.168.0.{random.randint(1, 254)}"
        num_paginas = random.randint(2, 10)
        inicio = datetime.now() - timedelta(days=random.randint(0, 7))
        for j in range(num_paginas):
            timestamp = inicio + timedelta(minutes=random.randint(0, 60))
            pagina = random.choice(links)
            registros.append({'ip': ip, 'timestamp': timestamp, 'pagina_visitada': pagina})
    return pd.DataFrame(registros)

# Paso 2: Procesar logs
logs = simular_logs()
logs['timestamp'] = pd.to_datetime(logs['timestamp'])
logs['sesion'] = (logs.groupby('ip')['timestamp'].diff() > pd.Timedelta(minutes=30)).cumsum().fillna(0)

# Calcular la página más visitada y cuántas veces se visitó
pagina_frecuente = logs.groupby(['ip', 'sesion'])['pagina_visitada'].agg(lambda x: x.mode()[0] if not x.mode().empty else x.iloc[0])
visitas_frecuentes = logs.groupby(['ip', 'sesion', 'pagina_visitada']).size().reset_index(name='visitas')
visitas_frecuentes = visitas_frecuentes.sort_values('visitas', ascending=False).drop_duplicates(subset=['ip', 'sesion'])

# Agrupar por sesión
sesiones = logs.groupby(['ip', 'sesion']).agg(
    duracion=('timestamp', lambda x: (x.max() - x.min()).seconds),
    paginas_visitadas=('pagina_visitada', 'count')
).reset_index()

# Añadir página principal y visitas
sesiones['pagina_principal'] = pagina_frecuente.values
sesiones = pd.merge(sesiones, visitas_frecuentes[['ip', 'sesion', 'visitas']], on=['ip', 'sesion'], how='left')

# Paso 3: Clustering
X = sesiones[['duracion', 'paginas_visitadas']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=3, random_state=42)
sesiones['cluster'] = kmeans.fit_predict(X_scaled)

# Paso 4: Mostrar resultados
print("\n📊 Promedios por clúster:")
print(sesiones.groupby('cluster')[['duracion', 'paginas_visitadas', 'visitas']].mean())

# Visualización detallada
plt.figure(figsize=(12, 9))
colors = ['red', 'green', 'blue']
for cluster in sesiones['cluster'].unique():
    datos = sesiones[sesiones['cluster'] == cluster]
    plt.scatter(datos['duracion'], datos['paginas_visitadas'],
                color=colors[cluster], label=f'Cluster {cluster}', s=70)
    for _, row in datos.iterrows():
        label = (
            f"{row['ip']}\n"
            f"{row['pagina_principal'][:35]}...\n"
            f"🔁 {int(row['visitas'])} visitas"
        )
        plt.annotate(label, (row['duracion'], row['paginas_visitadas']),
                     fontsize=7.5, alpha=0.7)

plt.xlabel("Duración de sesión (segundos)")
plt.ylabel("Páginas visitadas")
plt.title("🧠 Clustering de sesiones web simuladas (con visitas por página)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
