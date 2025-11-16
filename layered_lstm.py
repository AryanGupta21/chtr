import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns

from torch.optim.lr_scheduler import ReduceLROnPlateau


class MelDataset(Dataset):
    def __init__(self, df, mel_dir="mel"):
        self.df = df.reset_index(drop=True)
        self.mel_dir = mel_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = os.path.join(self.mel_dir, f"{os.path.splitext(row['filename'])[0]}_mel.pt")
        mel_tensor = torch.load(path)
        label = torch.tensor(row[["crackle", "wheeze"]].values.astype(float))
        return mel_tensor, label

class StackedLSTM(nn.Module):
    def __init__(self, input_size=128, hidden_size=64, output_size=2, num_layers=3):
        super(StackedLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3 if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.squeeze(1)  # [batch, 128, 128]
        x = x.permute(0, 2, 1)  # [batch, time=128, features=128]
        out, _ = self.lstm(x)
        last_step = out[:, -1, :]
        return self.fc(last_step)

def compute_metrics(y_true, y_pred):
    y_pred_bin = (y_pred > 0.5).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred_bin, average=None, zero_division=0)
    confmat_crackle = confusion_matrix(y_true[:, 0], y_pred_bin[:, 0])
    confmat_wheeze = confusion_matrix(y_true[:, 1], y_pred_bin[:, 1])
    return precision, recall, f1, confmat_crackle, confmat_wheeze

def train_epoch(model, loader, loss_fn, optimizer, device):
    model.train()
    total_loss = 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        outputs = model(X)
        loss = loss_fn(outputs, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X.size(0)
    return total_loss / len(loader.dataset)

def eval_model(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0
    preds, trues = [], []
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            loss = loss_fn(outputs, y)
            total_loss += loss.item() * X.size(0)
            preds.append(torch.sigmoid(outputs).cpu().numpy())
            trues.append(y.cpu().numpy())
    avg_loss = total_loss / len(loader.dataset)
    y_pred = np.vstack(preds)
    y_true = np.vstack(trues)
    precision, recall, f1, confmat_crackle, confmat_wheeze = compute_metrics(y_true, y_pred)
    return avg_loss, precision, recall, f1, confmat_crackle, confmat_wheeze

def plot_metrics(precision, recall, train_loss, val_loss, f1, confmat_crackle, confmat_wheeze,
                 test_metrics=None, confmat_crackle_test=None, confmat_wheeze_test=None):
    epochs = range(1, len(train_loss) + 1)
    fig, axs = plt.subplots(2, 4, figsize=(24, 10))

    axs[0, 0].plot(epochs, precision, marker='o', label='Val')
    axs[0, 0].set_title('Precision (avg)')
    axs[0, 0].set_xlabel('Epoch')
    axs[0, 0].set_ylabel('Precision')
    if test_metrics is not None:
        axs[0, 0].hlines(test_metrics[0], 1, len(train_loss), colors='r', linestyles='dashed', label='Test')
    axs[0, 0].legend()

    axs[0, 1].plot(epochs, recall, marker='o', label='Val')
    axs[0, 1].set_title('Recall (avg)')
    axs[0, 1].set_xlabel('Epoch')
    axs[0, 1].set_ylabel('Recall')
    if test_metrics is not None:
        axs[0, 1].hlines(test_metrics[1], 1, len(train_loss), colors='r', linestyles='dashed', label='Test')
    axs[0, 1].legend()

    axs[0, 2].plot(epochs, f1, marker='o', label='Val')
    axs[0, 2].set_title('F1 Score (avg)')
    axs[0, 2].set_xlabel('Epoch')
    axs[0, 2].set_ylabel('F1 Score')
    if test_metrics is not None:
        axs[0, 2].hlines(test_metrics[2], 1, len(train_loss), colors='r', linestyles='dashed', label='Test')
    axs[0, 2].legend()

    axs[0, 3].axis('off')

    axs[1, 0].plot(epochs, train_loss, marker='o', color='tab:orange')
    axs[1, 0].set_title('Train Loss')
    axs[1, 0].set_xlabel('Epoch')
    axs[1, 0].set_ylabel('Loss')

    axs[1, 1].plot(epochs, val_loss, marker='o', color='tab:green')
    axs[1, 1].set_title('Val Loss')
    axs[1, 1].set_xlabel('Epoch')
    axs[1, 1].set_ylabel('Loss')

    sns.heatmap(confmat_crackle, annot=True, fmt='d', cmap='Blues', ax=axs[1, 2])
    axs[1, 2].set_title('Confusion Matrix Crackle')

    sns.heatmap(confmat_wheeze, annot=True, fmt='d', cmap='Greens', ax=axs[1, 3])
    axs[1, 3].set_title('Confusion Matrix Wheeze')

    plt.tight_layout()
    plt.show()
    plt.close('all')  # Closing plot window to let script continue

    # Optional: plot test confusion matrices separately
    if confmat_crackle_test is not None and confmat_wheeze_test is not None:
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))
        sns.heatmap(confmat_crackle_test, annot=True, fmt='d', cmap='Blues', ax=axs[0])
        axs[0].set_title('Test Conf Matrix Crackle')
        sns.heatmap(confmat_wheeze_test, annot=True, fmt='d', cmap='Greens', ax=axs[1])
        axs[1].set_title('Test Conf Matrix Wheeze')
        plt.tight_layout()
        plt.show()
        plt.close('all')

# Main script
df = pd.read_csv("icbhi.csv")

train_val_df, test_df = train_test_split(df, test_size=0.15, random_state=42)
train_df, val_df = train_test_split(train_val_df, test_size=0.15, random_state=43)

train_dataset = MelDataset(train_df)
val_dataset = MelDataset(val_df)
test_dataset = MelDataset(test_df)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = StackedLSTM(num_layers=3).to(device)

loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 100

precision_list = []
recall_list = []
train_loss_list = []
val_loss_list = []
f1_list = []
confmat_final_crackle = None
confmat_final_wheeze = None


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
            
early_stopping = EarlyStopping(patience=5, min_delta=1e-4)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)


for epoch in range(num_epochs):
    train_loss = train_epoch(model, train_loader, loss_fn, optimizer, device)
    val_loss, precision, recall, f1, confmat_crackle, confmat_wheeze = eval_model(model, val_loader, loss_fn, device)

    train_loss_list.append(train_loss)
    val_loss_list.append(val_loss)
    precision_list.append(precision.mean())
    recall_list.append(recall.mean())
    f1_list.append(f1.mean())
    confmat_final_crackle = confmat_crackle
    confmat_final_wheeze = confmat_wheeze

    print(f"Epoch {epoch+1}/{num_epochs}")
    print(f"Train Loss: {train_loss:.4f} Val Loss: {val_loss:.4f}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    print(f"Confusion Matrix crackle:\n{confmat_crackle}")
    print(f"Confusion Matrix wheeze:\n{confmat_wheeze}")

    scheduler.step(val_loss)

    early_stopping(val_loss)
    if early_stopping.early_stop:
        print(f"Early stopping triggered at epoch {epoch+1}")
        break
    
    
    
# Test evaluation after training completed
print("Starting test evaluation...", flush=True)
test_loss, precision_test, recall_test, f1_test, confmat_crackle_test, confmat_wheeze_test = eval_model(
    model, test_loader, loss_fn, device)
print("Test evaluation done.", flush=True)

print("\nTest Set Metrics:", flush=True)
print(f"Test Loss: {test_loss:.4f}", flush=True)
print(f"Test Precision: {precision_test}", flush=True)
print(f"Test Recall: {recall_test}", flush=True)
print(f"Test F1 Score: {f1_test}", flush=True)
print(f"Test Confusion Matrix (crackle):\n{confmat_crackle_test}", flush=True)
print(f"Test Confusion Matrix (wheeze):\n{confmat_wheeze_test}", flush=True)

print("Starting plotting test results...", flush=True)
plot_metrics(precision_list, recall_list, train_loss_list, val_loss_list, f1_list,
            confmat_final_crackle, confmat_final_wheeze,
            test_metrics=[precision_test.mean(), recall_test.mean(), f1_test.mean()],
            confmat_crackle_test=confmat_crackle_test,
            confmat_wheeze_test=confmat_wheeze_test)
print("Done plotting.", flush=True)