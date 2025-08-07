#  Handwritten Digit Recognition with ResNet-10

Ce projet implémente un modèle **ResNet-10** entraîné sur le dataset **MNIST** pour reconnaître des chiffres manuscrits (0 à 9).  
Le modèle est construit avec **TensorFlow/Keras** et atteint une précision élevée grâce à l’utilisation de blocs résiduels.

---

##  Dataset

- **MNIST** : 60 000 images pour l'entraînement, 10 000 pour le test
- Images : 28x28 pixels, niveaux de gris
- Classes : 10 (chiffres de 0 à 9)

---

##  Architecture du modèle : ResNet-10

Le modèle utilisé est basé sur l'architecture **ResNet-10**, une version légère des réseaux résiduels.

| Composant         | Description                                                              |
|-------------------|---------------------------------------------------------------------------|
| **Input Layer**    | 28x28x1 (image en niveaux de gris)                                       |
| **Conv2D + BN + ReLU** | Première couche convolutionnelle avec normalisation et activation        |
| **Residual Block x2** | 2 blocs résiduels avec 64 filtres                                      |
| **Residual Block x2** | 2 blocs résiduels avec 128 filtres + downsampling                      |
| **GlobalAvgPool**  | Réduction de dimensions par moyennage global                             |
| **Dense (Softmax)**| Couche finale pour la classification à 10 classes                        |

---

##  Pourquoi ResNet-10 ?

| Avantage                         | Explication                                                                 |
|----------------------------------|-----------------------------------------------------------------------------|
| ✅ **Blocs résiduels**           | Facilitent l'apprentissage de réseaux profonds en évitant le gradient vanishing |
| ✅ **Architecture légère**       | Moins de couches que ResNet-50 ou 101, donc plus rapide à entraîner/tester |
| ✅ **Bonne précision sur MNIST** | Suffisamment puissant pour ce dataset sans overfitting                     |

---

##  Entraînement du modèle

```python
model = build_resnet10()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train_cat, epochs=5, validation_split=0.2)
