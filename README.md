# Handwritten Digit Recognition with ResNet-10

Ce projet implémente un modèle **ResNet-10** entraîné sur le dataset **MNIST** pour reconnaître des chiffres manuscrits (de 0 à 9).
Le modèle est construit à l'aide de **TensorFlow/Keras** et atteint une précision élevée grâce à l’utilisation de blocs résiduels.

---

## Dataset

- **MNIST** : 60 000 images pour l'entraînement, 10 000 pour le test
- Format des images : 28x28 pixels, en niveaux de gris
- Classes : 10 (chiffres de 0 à 9)

---

## Architecture du modèle : ResNet-10

Le modèle repose sur l'architecture **ResNet-10**, une version compacte des réseaux résiduels, bien adaptée aux tâches simples comme MNIST.

| Composant            | Description                                                               |
|----------------------|----------------------------------------------------------------------------|
| **Input Layer**      | Entrée 28x28x1 (image en niveaux de gris)                                 |
| **Conv2D + BN + ReLU** | Première couche convolutionnelle avec normalisation et activation ReLU   |
| **Residual Block x2**| 2 blocs résiduels avec 64 filtres                                          |
| **Residual Block x2**| 2 blocs résiduels avec 128 filtres (avec réduction de dimensions)          |
| **Global Avg Pool**  | Moyennage global pour réduire la dimension                                 |
| **Dense (Softmax)**  | Couche de sortie pour la classification sur 10 classes                    |

---

## Pourquoi ResNet-10 ?

| Avantage                       | Explication                                                                 |
|--------------------------------|------------------------------------------------------------------------------|
| **Blocs résiduels**           | Facilitent l'entraînement en permettant une meilleure propagation du gradient |
| **Architecture légère**       | Moins de couches que ResNet-50 ou ResNet-101, donc plus rapide               |
| **Efficace sur MNIST**        | Suffisamment puissant sans surajuster ce dataset simple                      |

---

## Entraînement du modèle

```python
model = build_resnet10()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train_cat, epochs=5, validation_split=0.2)
```

Le modèle est sauvegardé après l'entraînement sous le nom `cnn_m.h5`.

---

## Prédiction de chiffres manuscrits

L'application permet à l'utilisateur de charger une image contenant un chiffre manuscrit pour prédiction.

1. L'utilisateur importe une image depuis son ordinateur.
2. L'image est convertie en niveaux de gris, redimensionnée à 28x28 pixels, puis prétraitée (binarisation, inversion si nécessaire).
3. L'image prétraitée est passée au modèle entraîné (CNN) pour prédire le chiffre.
4. Le chiffre prédit est affiché à l'utilisateur avec les images originale et prétraitée.

---

## Tests sur des exemples réels

### 1. Chiffre **2** manuscrit (couleur noire)

**Entrée** :
![number2](https://github.com/user-attachments/assets/a523d081-b81f-4e99-b294-556031ecbfc2)

**Résultat** :
<img width="400" height="188" alt="pred2" src="https://github.com/user-attachments/assets/09d544ec-8c00-450a-821e-744ab4895149" />

---

### 2. Chiffre **7** manuscrit (couleur noire)

**Entrée** :
![number7](https://github.com/user-attachments/assets/b8a6c49f-7f30-41e0-949a-06a745fc8910)

**Résultat** :
<img width="303" height="172" alt="pred7" src="https://github.com/user-attachments/assets/4075f58e-8a86-46a7-9c50-63431d94ea37" />

---

### 3. Chiffre **4** manuscrit (couleur bleue)

**Entrée** :
![number4](https://github.com/user-attachments/assets/f7c61081-245c-40c8-8174-330b62aab024)

**Résultat** :
<img width="330" height="185" alt="pred4" src="https://github.com/user-attachments/assets/f1911bc3-b5f9-470c-9c8f-8dff954f3e3f" />

---
