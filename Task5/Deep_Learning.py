import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

project_path = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

dataset_path = os.path.join(project_path, "Dataset")
figures_path = os.path.join(project_path, "Figures")
os.makedirs(figures_path, exist_ok=True)

file_path = os.path.join(dataset_path, "AirQualityUCI_cleaned.csv")
df = pd.read_csv(file_path)


print("1. DATASET LOADED FOR DEEP LEARNING")

print(f"Dataset shape: {df.shape}")

print("2. FEATURE ENGINEERING")


df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
df["Time_dt"] = pd.to_datetime(df["Time"], format="%H:%M:%S", errors="coerce")

df["Hour"] = df["Time_dt"].dt.hour
df["Month"] = df["Date"].dt.month
df["DayOfWeek"] = df["Date"].dt.dayofweek

target = "CO(GT)"
features = [
    "C6H6(GT)", "NOx(GT)", "NO2(GT)",
    "PT08.S1(CO)", "PT08.S2(NMHC)", "PT08.S3(NOx)",
    "PT08.S4(NO2)", "PT08.S5(O3)",
    "T", "RH", "AH", "Hour", "Month", "DayOfWeek"
]

data = df[features + [target]].dropna(subset=[target]).copy()

X = data[features].values
y = data[target].values

X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
    X, y, test_size=0.20, random_state=42
)

X_train_raw, X_val_raw, y_train_raw, y_val_raw = train_test_split(
    X_train_raw, y_train_raw, test_size=0.20, random_state=42
)
imputer = SimpleImputer(strategy="median")
X_train_imp = imputer.fit_transform(X_train_raw)
X_val_imp = imputer.transform(X_val_raw)
X_test_imp = imputer.transform(X_test_raw)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imp)
X_val_scaled = scaler.transform(X_val_imp)
X_test_scaled = scaler.transform(X_test_imp)

X_train = torch.FloatTensor(X_train_scaled)
y_train = torch.FloatTensor(y_train_raw).view(-1, 1)
X_val = torch.FloatTensor(X_val_scaled)
y_val = torch.FloatTensor(y_val_raw).view(-1, 1)
X_test = torch.FloatTensor(X_test_scaled)
y_test = torch.FloatTensor(y_test_raw).view(-1, 1)

print(f"Training samples: {X_train.shape[0]}, Features: {X_train.shape[1]}")
print(f"Validation samples: {X_val.shape[0]}")
print(f"Testing samples: {X_test.shape[0]}")

train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

val_dataset = TensorDataset(X_val, y_val)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)


print("3. BUILDING DEEP NEURAL NETWORK (PYTORCH)")


torch.manual_seed(42)

class AirQualityNN(nn.Module):
    def __init__(self, input_dim):
        super(AirQualityNN, self).__init__()
        self.network = nn.Sequential(
    
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.network(x)

input_dim = X_train.shape[1]
model = AirQualityNN(input_dim)
print(model)

print("4. TRAINING DEEP LEARNING MODEL")


criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.005)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

epochs = 120
patience = 15

history = {'loss': [], 'val_loss': [], 'mae': [], 'val_mae': []}

best_val_loss = float('inf')
best_model_weights = None
epochs_no_improve = 0

for epoch in range(epochs):

    model.train()
    train_loss, train_mae_sum = 0.0, 0.0
    
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        predictions = model(batch_X)
        loss = criterion(predictions, batch_y)
        
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() * batch_X.size(0)
        train_mae_sum += torch.abs(predictions - batch_y).sum().item()
        
    train_loss = train_loss / len(train_loader.dataset)
    train_mae = train_mae_sum / len(train_loader.dataset)
    
    model.eval()
    val_loss, val_mae_sum = 0.0, 0.0
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            val_loss += loss.item() * batch_X.size(0)
            val_mae_sum += torch.abs(predictions - batch_y).sum().item()
            
    val_loss = val_loss / len(val_loader.dataset)
    val_mae = val_mae_sum / len(val_loader.dataset)
    
    history['loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['mae'].append(train_mae)
    history['val_mae'].append(val_mae)
    
    scheduler.step(val_loss)
    
    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"Epoch [{epoch+1}/{epochs}] | Train Loss (MSE): {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val MAE: {val_mae:.4f}")
  
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_weights = copy.deepcopy(model.state_dict())
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f"\nEarly stopping triggered at epoch {epoch+1}!")
            break

model.load_state_dict(best_model_weights)
print("Restored best model weights.")

print("5. MODEL EVALUATION")


model.eval()
with torch.no_grad():
    dl_predictions = model(X_test).numpy().flatten()

y_test_np = y_test.numpy().flatten()

dl_mae = mean_absolute_error(y_test_np, dl_predictions)
dl_rmse = np.sqrt(mean_squared_error(y_test_np, dl_predictions))
dl_r2 = r2_score(y_test_np, dl_predictions)

print(f"Deep Learning Test MAE:  {dl_mae:.4f}")
print(f"Deep Learning Test RMSE: {dl_rmse:.4f}")
print(f"Deep Learning Test R²:   {dl_r2:.4f}")

# Save metrics to CSV
dl_eval_path = os.path.join(dataset_path, "Deep_Learning_Evaluation.csv")
eval_df = pd.DataFrame([{
    "Model": "Deep Neural Network (PyTorch)",
    "MAE": dl_mae,
    "RMSE": dl_rmse,
    "R2": dl_r2
}])
eval_df.to_csv(dl_eval_path, index=False)


# Figure 1: Learning Curves (Loss & MAE over Epochs)
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history["loss"], label="Training Loss (MSE)")
plt.plot(history["val_loss"], label="Validation Loss (MSE)", linestyle="--")
plt.title("Loss Progression Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Mean Squared Error")
plt.legend()
plt.grid(True, linestyle=":", alpha=0.6)

plt.subplot(1, 2, 2)
plt.plot(history["mae"], label="Training MAE")
plt.plot(history["val_mae"], label="Validation MAE", linestyle="--")
plt.title("MAE Progression Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Mean Absolute Error")
plt.legend()
plt.grid(True, linestyle=":", alpha=0.6)

plt.tight_layout()
plt.savefig(os.path.join(figures_path, "dl_training_curves.png"))
plt.show()

# Figure 2: Actual vs Predicted
plt.figure(figsize=(8, 6))
plt.scatter(y_test_np, dl_predictions, alpha=0.4, color="#2b5c8f")
mn = min(y_test_np.min(), dl_predictions.min())
mx = max(y_test_np.max(), dl_predictions.max())
plt.plot([mn, mx], [mn, mx], color="red", linestyle="--", label="Ideal 1:1 Line")
plt.xlabel("Actual CO Concentration")
plt.ylabel("Predicted CO Concentration")
plt.title("Deep Learning: Actual vs Predicted CO")
plt.legend()
plt.grid(True, linestyle=":", alpha=0.6)
plt.tight_layout()
plt.savefig(os.path.join(figures_path, "dl_actual_vs_predicted.png"))
plt.show()

# Figure 3: Residual Distribution
residuals = y_test_np - dl_predictions
plt.figure(figsize=(8, 6))
plt.scatter(dl_predictions, residuals, alpha=0.4, color="#2b5c8f")
plt.axhline(0, color="red", linestyle="--")
plt.xlabel("Predicted CO Concentration")
plt.ylabel("Residuals (Actual - Predicted)")
plt.title("Deep Learning: Residual Analysis")
plt.grid(True, linestyle=":", alpha=0.6)
plt.tight_layout()
plt.savefig(os.path.join(figures_path, "dl_residual_analysis.png"))
plt.show()
