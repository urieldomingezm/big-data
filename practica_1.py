import requests
from bs4 import BeautifulSoup
import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from wordcloud import WordCloud

# 1. Extraer texto desde una URL
def extraer_texto(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    parrafos = soup.find_all('p')
    texto = [p.get_text().strip() for p in parrafos if len(p.get_text()) > 40]
    return texto

# URL para probar
url = "https://elpais.com/tecnologia/"
textos = extraer_texto(url)

# 2. Crear DataFrame y simular sentimientos (positivos o negativos)
data = pd.DataFrame({
    'texto': textos,
    'sentimiento': [random.choice([0, 1]) for _ in textos]
})

# 3. Preprocesamiento
vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(data['texto'])
y = data['sentimiento']

# 4. Entrenamiento
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = MultinomialNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 5. Visualizar matriz de confusión
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=["Negativo", "Positivo"],
            yticklabels=["Negativo", "Positivo"])
plt.title("🔍 Matriz de Confusión del Modelo")
plt.xlabel("Predicción")
plt.ylabel("Real")
plt.tight_layout()
plt.show()

# 6. Gráfica de pastel de sentimientos
plt.figure(figsize=(5, 5))
data['sentimiento'].replace({1: 'Positivo', 0: 'Negativo'}).value_counts().plot.pie(
    autopct='%1.1f%%', startangle=90, colors=['lightgreen', 'lightcoral'])
plt.title("📊 Distribución de Sentimientos (Simulados)")
plt.ylabel("")
plt.tight_layout()
plt.show()

# 7. Mostrar predicciones de prueba
print("\n🧪 Ejemplos de predicciones:")
for i in range(5):
    texto_corto = data['texto'].iloc[i][:100].replace('\n', ' ')
    etiqueta = y.iloc[i]
    pred = model.predict(vectorizer.transform([data['texto'].iloc[i]]))[0]
    color = "🟢" if pred == 1 else "🔴"
    print(f"{color} {'Positivo' if pred else 'Negativo'} | Real: {'✅' if etiqueta == pred else '❌'}")
    print(f"» {texto_corto}...\n")

# 8. Mostrar palabras más usadas
word_counts = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
word_frequencies = word_counts.sum().sort_values(ascending=False)

plt.figure(figsize=(10, 6))
word_frequencies.head(20).plot(kind='barh', color='dodgerblue')
plt.title("🔑 Palabras Más Usadas")
plt.xlabel("Frecuencia")
plt.ylabel("Palabras")
plt.tight_layout()
plt.show()

# 9. Mostrar palabras clave asociadas a sentimientos
# Palabras asociadas a sentimiento positivo (1)
positive_words = [vectorizer.get_feature_names_out()[i] for i in model.feature_log_prob_[1].argsort()[-20:][::-1]]
# Palabras asociadas a sentimiento negativo (0)
negative_words = [vectorizer.get_feature_names_out()[i] for i in model.feature_log_prob_[0].argsort()[-20:][::-1]]

# Palabra más frecuente asociada a positivo/negativo
print(f"\n🔵 Palabras clave asociadas al sentimiento positivo:")
print(positive_words)

print(f"\n🔴 Palabras clave asociadas al sentimiento negativo:")
print(negative_words)

# 10. Generar nubes de palabras de los textos positivos y negativos
# Palabras clave positivas
positive_texts = ' '.join(data['texto'][data['sentimiento'] == 1])
negative_texts = ' '.join(data['texto'][data['sentimiento'] == 0])

# Crear la nube de palabras para textos positivos
positive_wordcloud = WordCloud(width=800, height=400, background_color='white').generate(positive_texts)
plt.figure(figsize=(10, 6))
plt.imshow(positive_wordcloud, interpolation='bilinear')
plt.title("🟢 Nube de Palabras para Sentimientos Positivos")
plt.axis('off')
plt.tight_layout()
plt.show()

# Crear la nube de palabras para textos negativos
negative_wordcloud = WordCloud(width=800, height=400, background_color='white').generate(negative_texts)
plt.figure(figsize=(10, 6))
plt.imshow(negative_wordcloud, interpolation='bilinear')
plt.title("🔴 Nube de Palabras para Sentimientos Negativos")
plt.axis('off')
plt.tight_layout()
plt.show()
