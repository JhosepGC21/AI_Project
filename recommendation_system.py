# Crea un sistema de recomendación simple en Python.
# Utiliza datos de productos y sus características.
# Entrena un modelo de clasificación y permite recomendar
# un producto a partir de sus características.
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Datos de ejemplo
data = {
    "precio": [10, 15, 20, 25, 30, 35, 40, 45],
    "popularidad": [2, 3, 4, 5, 6, 7, 8, 9],
    "categoria": [
        "A", "A", "A", "B",
        "B", "B", "C", "C"
    ]
}

df = pd.DataFrame(data)

# Características utilizadas para el modelo
features = ["precio", "popularidad"]

X = df[features]
y = df["categoria"]

# Separar datos para entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42
)

# Crear modelo
model = KNeighborsClassifier(n_neighbors=3)

# Entrenar modelo
model.fit(X_train, y_train)

# Realizar predicciones
predictions = model.predict(X_test)

# Evaluar modelo
accuracy = accuracy_score(y_test, predictions)

print("Precisión del modelo:", accuracy)

# Función para recomendar categoría
def recommend_product(precio, popularidad):
    prediction = model.predict([[precio, popularidad]])
    return prediction[0]


# Ejemplo de recomendación
recommended = recommend_product(28, 6)

print("Categoría recomendada:", recommended)