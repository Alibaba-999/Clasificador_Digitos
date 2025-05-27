# 🧠 MNIST Classifier - Reconocimiento de Dígitos con TensorFlow

## 📌 Descripción

Este proyecto utiliza **TensorFlow** para entrenar una red neuronal capaz de reconocer **dígitos escritos a mano** utilizando el conjunto de datos **MNIST**.

A través del entrenamiento supervisado, el modelo aprende a identificar números del **0 al 9** en imágenes y alcanza una **precisión del 97.7%** en datos de prueba.

---

## 🚀 Instalación y Configuración

### **Requisitos Previos**

Antes de ejecutar el proyecto, asegúrate de tener instalado:

- ✅ Python 3.11 o superior
- ✅ TensorFlow
- ✅ NumPy
- ✅ Matplotlib
- ✅ Docker *(opcional, para entorno de desarrollo aislado)*

### **Instalación de Dependencias**

Ejecuta el siguiente comando en la terminal para instalar todas las librerías necesarias:

```bash
pip install -r requirements.txt
```

---

## 🎯 Uso del Proyecto

### **Entrenar el Modelo**

Para iniciar el entrenamiento del modelo, ejecuta:

```bash
python mnist_classifier.py
```

El modelo aprenderá con el conjunto de datos **MNIST**, ajustará los parámetros y mostrará su **precisión** después de cada **época**.

### **Evaluación del Modelo**

Después del entrenamiento, el modelo se probará con nuevos datos. La precisión final aparecerá en pantalla, indicando qué tan bien reconoce dígitos que nunca ha visto antes.

### **Guardado del Modelo**

Al finalizar, el modelo entrenado se guardará automáticamente como `mnist_classifier.h5`, permitiendo reutilizarlo sin necesidad de volver a entrenar.

---

## 🏗 Arquitectura del Modelo

El modelo se compone de las siguientes capas en una estructura **secuencial**:

1️⃣ **Entrada:** Convierte imágenes **28×28 píxeles** en un vector de **784 valores**.

2️⃣ **Capa oculta:** Tiene **128 neuronas** con activación **ReLU**, lo que ayuda a identificar patrones.

3️⃣ **Capa dropout:** Apaga aleatoriamente **el 20% de las neuronas** en cada iteración para evitar sobreentrenamiento.

4️⃣ **Capa de salida:** Usa **Softmax**, asignando una probabilidad a **cada dígito (0-9)**.

---

## 📊 Resultados y Evaluación

🔹 **Entrenamiento en 5 épocas:** El modelo alcanza **97.7% de precisión** en datos de prueba.

---

## 🗄 Licencia

Este proyecto es **código abierto (open source)** y está disponible para uso libre.
