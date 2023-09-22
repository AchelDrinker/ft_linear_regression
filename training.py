import csv, json
import matplotlib.pyplot as plt
import numpy as np

# Lecture des paramètres sauvegardés s'ils existent sinon initialisation
try:
    with open('params.json', 'r') as f:
        params = json.load(f)
        theta0 = params['theta0']
        theta1 = params['theta1']
except FileNotFoundError:
    theta0 = 0.0  # Valeur initiale de theta0
    theta1 = 0.0  # Valeur initiale de theta1

# Chargement des données depuis le fichier CSV
X = []
y = []
with open('./data.csv', 'r') as file:
    reader = csv.reader(file)
    next(reader)  # Ignorer l'en-tête
    for row in reader:
        X.append(float(row[0]))
        y.append(float(row[1]))

# Conversion des listes en tableaux NumPy pour les calculs
X_np = np.array(X)
y_np = np.array(y)

# Normalisation des données Z-score
# x' = (x - moyenne) / ecart-type
X_mean = np.mean(X_np) # Moyenne
X_std = np.std(X_np) # Écart-type
y_mean = np.mean(y_np) # Moyenne
y_std = np.std(y_np) # Écart-type
X_normalized = (X_np - X_mean) / X_std # Normalisation
y_normalized = (y_np - y_mean) / y_std # Normalisation

# Initialisation des paramètres pour l'entraînement
learning_rate = 0.5
num_iterations = 1000000000
previous_error = float('inf')
error_threshold = 1e-4
early_stopping_threshold = 1e-5
learning_rate_decay = 0.9

# Entraînement du modèle
for i in range(num_iterations):
    y_pred_normalized = theta0 + theta1 * X_normalized
    error = y_pred_normalized - y_normalized
    current_error = np.mean(error**2)
    
    # Arrêt anticipé si l'erreur a peu changé
    if abs(previous_error - current_error) < early_stopping_threshold:
        print("Early stopping triggered. Stopping training.")
        break
    
    # Ajustement du taux d'apprentissage
    if abs(previous_error - current_error) < error_threshold:
        learning_rate *= learning_rate_decay
    
    # Mise à jour de l'erreur précédente
    previous_error = current_error
    
    # Calcul des gradients et mise à jour des paramètres
    grad_theta0 = np.mean(error) # Dérivée partielle par rapport à theta0
    grad_theta1 = np.mean(error * X_normalized) # Dérivée partielle par rapport à theta1
    theta0 -= learning_rate * grad_theta0
    theta1 -= learning_rate * grad_theta1
    
    # Vérification de la divergence
    if np.isnan(theta0) or np.isnan(theta1):
        print("Divergence detected. Stopping training.")
        break

# Dénormalisation des paramètres
theta0_denormalized = theta0 * y_std + y_mean - (theta1 * X_mean * y_std / X_std)
theta1_denormalized = theta1 * y_std / X_std

# Calcul des prédictions avec les paramètres dénormalisés
y_pred = [theta0_denormalized + theta1_denormalized * x for x in X]

# Sauvegarde des paramètres
with open('params.json', 'w') as f:
    json.dump({'theta0': theta0_denormalized, 'theta1': theta1_denormalized}, f)

# Calcul et affichage des métriques de performance
print(f"Theta0 : {theta0_denormalized}, Theta1 : {theta1_denormalized}")

# ----------------------------------------------------------------------------------------------------------------------

# MSE (Mean Squared Error)
# Il s'agit de l'erreur quadratique moyenne.
# C'est une mesure qui indique comment les prédictions du modèle se rapprochent des valeurs réelles.
# Plus cette valeur est petite, mieux c'est (bien que d'autres métriques soient également importantes).
mse = np.mean((np.array(y) - np.array(y_pred))**2)
print(f"Erreur Quadratique Moyenne (MSE) : {mse}")

# ----------------------------------------------------------------------------------------------------------------------

# RMSE (Root Mean Squared Error)
# C'est la racine carrée de l'erreur quadratique moyenne.
# Elle est utile car elle est dans les mêmes unités que la variable cible (y).
# Elle donne également une idée de l'erreur que vous pouvez attendre dans les prédictions.
rmse = np.sqrt(mse)
print(f"Racine de l'Erreur Quadratique Moyenne (RMSE) : {rmse}")

# ----------------------------------------------------------------------------------------------------------------------

# R² (Coefficient de Détermination)
# Il s'agit d'une mesure statistique qui indique la proportion de la variance
# de la variable dépendante qui est prévisible à partir des variables indépendantes.
# En d'autres termes, il indique à quel point votre modèle explique les variations dans les données.
# Le R² varie entre 0 et 1, où un R² proche de 1 indique que le modèle explique une grande partie de la variance.
ss_res = np.sum((np.array(y) - np.array(y_pred))**2) # Somme des carrés des résidus (différences entre les valeurs observées et les valeurs prédites)
ss_tot = np.sum((np.array(y) - np.mean(y))**2) # Somme des carrés des différences entre les valeurs observées et leur moyenne
r2 = 1 - (ss_res / ss_tot)
print(f"Coefficient de Détermination (R²) : {r2}")

# ----------------------------------------------------------------------------------------------------------------------

# Affichage graphique des résultats
plt.scatter(X, y, color='blue', label='Data points')
plt.plot(X, y_pred, color='red', label='Regression line')
plt.xlabel('Kilométrage')
plt.ylabel('Prix')
plt.title('Régression linéaire')
plt.legend()
plt.show()
