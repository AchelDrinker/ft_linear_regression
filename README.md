# 📊 Simple Linear Regression in Python

## 📝 Description

Ce projet est une implémentation simple mais efficace d'une **régression linéaire** en Python. Utilisant des bibliothèques de premier plan comme `numpy` pour les calculs numériques et `matplotlib` pour la visualisation, ce projet offre un excellent point de départ pour quiconque s'intéresse au Machine Learning.

---

## 🛠 Prérequis

Pour utiliser ce projet, vous aurez besoin de :

- Python 3.x
- NumPy
- Matplotlib
- Un fichier CSV pour les données

---

## 🚀 Installation

### Clonez le dépôt

```
git clone https://github.com/AchelDrinker/ft_linear_regression.git
```

### Installez les dépendances

Ouvrez un terminal et exécutez la commande suivante :

```
pip install numpy matplotlib
```

---

## 🎯 Utilisation

1. **Préparation des données** : Placez votre fichier CSV dans le même répertoire que le script principal. Assurez-vous que le fichier contient deux colonnes, l'une pour la variable indépendante et l'autre pour la variable dépendante.

2. **Exécution du script** :

```
python training.py
```

3. **Résultats** : Le programme effectue la régression linéaire, sauvegarde les paramètres du modèle dans un fichier `params.json` et affiche diverses métriques d'évaluation.

4. **Visualisation** : Une représentation graphique des données et de la ligne de régression est également affichée.

5. **Prédiction** : Un script est également disponible pour prédire la variable dépendante en fonction d'une variable indépendante que vous ajoutez en prompt.

6. **Exécution du script** :
   ```
   python predict.py
   ```

---

## 📈 Métriques de performance

Ce projet fournit les métriques de performance suivantes :

- **Erreur Quadratique Moyenne (MSE)**
- **Racine de l'Erreur Quadratique Moyenne (RMSE)**
- **Coefficient de Détermination (R²)**

---

## 📚 Sources
Pour réaliser ce projet, je me suis appuyé sur les ressources suivantes :

Documentation NumPy : [Site officiel](https://numpy.org/doc/)

Matplotlib User Guide : [Documentation](https://matplotlib.org/stable/users/index.html)

Introduction à la Régression Linéaire : [Tutoriel](https://www.youtube.com/playlist?list=PLO_fdPEVlfKqUF5BPKjGSh7aV9aBshrpY)

Cours de Machine Learning : [Cours](https://github.com/AchelDrinker/ft_linear_regression/blob/b7bed1a2e93af3d0f90375b98e79e193e2c02f8b/Apprendre_le_ML_en_une_semaine.pdf)https://github.com/AchelDrinker/ft_linear_regression/blob/b7bed1a2e93af3d0f90375b98e79e193e2c02f8b/Apprendre_le_ML_en_une_semaine.pdf
