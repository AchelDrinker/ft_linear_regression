import json

def estimate_price(mileage, theta0, theta1, X_mean, X_std, y_mean, y_std):
    mileage_normalized = (mileage - X_mean) / X_std
    price_normalized = theta0 + (theta1 * mileage_normalized)
    return (price_normalized * y_std) + y_mean

# Définit des valeurs par défaut
theta0_optimized = 0.0
theta1_optimized = 0.0
X_mean = 0.0
X_std = 1.0
y_mean = 0.0
y_std = 1.0

# Essaye de lire les valeurs à partir du fichier et les assigne aux variables
try:
    with open('params.json', 'r') as f:
        params = json.load(f)
        theta0_optimized = params.get('theta0', 0.0)
        theta1_optimized = params.get('theta1', 0.0)
        X_mean = params.get('X_mean', 0.0)
        X_std = params.get('X_std', 1.0)
        y_mean = params.get('y_mean', 0.0)
        y_std = params.get('y_std', 1.0)
except (FileNotFoundError, json.JSONDecodeError):
    pass

# Demande le kilométrage à l'utilisateur
mileage = float(input("Veuillez entrer le kilométrage: "))
price = estimate_price(mileage, theta0_optimized, theta1_optimized, X_mean, X_std, y_mean, y_std)
price_rounded = round(price, 2)
print(f"Le prix estimé est: {price_rounded}")
