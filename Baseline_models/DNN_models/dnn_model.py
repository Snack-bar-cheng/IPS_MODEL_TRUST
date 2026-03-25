"""
PyTorch DNN 回归模型（支持 GPU/MPS）

该模块将 notebook 中的训练流程封装为可在 baseline 管线中调用的
sklearn 风格模型。默认自动选择设备：CUDA > MPS > CPU。
"""

from __future__ import annotations

import copy
import random
import time
import os
import sys
from typing import List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler



def _set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _select_device(preference: str = "auto") -> torch.device:
    """
    Select compute device with priority: CUDA > MPS > CPU by default.
    preference: 'auto' | 'cuda' | 'mps' | 'cpu'
    """
    pref = preference.lower()
    if pref == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if pref == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if pref == "cpu":
        return torch.device("cpu")

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class _RegressionNet(nn.Module):
    """Feed-forward network used by DNNRegressor."""

    def __init__(self, input_size: int, hidden_layers: Sequence[int], dropout: float, use_batch_norm: bool):
        super().__init__()
        layers: List[nn.Module] = []
        prev = input_size
        for idx, hidden in enumerate(hidden_layers):
            layers.append(nn.Linear(prev, hidden))
            if use_batch_norm and idx < len(hidden_layers) - 1:
                layers.append(nn.BatchNorm1d(hidden))
            layers.append(nn.ReLU())
            if dropout > 0 and idx < len(hidden_layers) - 1:
                layers.append(nn.Dropout(dropout))
            prev = hidden
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(x).squeeze()


class DNNRegressor(BaseEstimator, RegressorMixin):
    """
    sklearn 风格的 DNN 回归器，内部使用 PyTorch，支持 GPU/MPS。
    设计用于与现有 baseline 管线保持一致（fit / predict 接口）。
    """

    def __init__(
        self,
        hidden_layers: Sequence[int] = (512, 443, 374, 306, 237, 169, 100, 32),
        dropout: float = 0.2,
        use_batch_norm: bool = True,
        batch_size: int = 128,
        learning_rate: float = 1e-3,
        max_epochs: int = 2000,
        early_stopping_patience: int = 30,
        validation_fraction: float = 0.15,
        weight_decay: float = 1e-5,
        device_preference: str = "auto",
        random_seed: int = 1,
        verbose: bool = False,
    ):
        self.hidden_layers = tuple(hidden_layers)
        self.dropout = dropout
        self.use_batch_norm = use_batch_norm
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.early_stopping_patience = early_stopping_patience
        self.validation_fraction = validation_fraction
        self.weight_decay = weight_decay
        self.device_preference = device_preference
        self.random_seed = random_seed
        self.verbose = verbose

        # runtime attributes
        self.device: torch.device = _select_device(device_preference)
        self.model: Optional[_RegressionNet] = None
        self.scaler: Optional[StandardScaler] = None
        self.input_size: Optional[int] = None
        self._is_fitted: bool = False

    # -------- sklearn API --------
    def fit(self, X: np.ndarray, y: np.ndarray):
        _set_seed(self.random_seed)
        self.device = _select_device(self.device_preference)

        X_np = np.asarray(X, dtype=np.float32)
        y_np = np.asarray(y, dtype=np.float32).ravel()
        self.input_size = X_np.shape[1]

        # scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_np).astype(np.float32)

        # split validation set
        val_size = int(len(X_scaled) * self.validation_fraction)
        if val_size < 1:
            val_size = max(1, len(X_scaled) // 10)  # keep small validation set
        indices = np.random.permutation(len(X_scaled))
        val_idx = indices[:val_size]
        train_idx = indices[val_size:]

        X_train, y_train = X_scaled[train_idx], y_np[train_idx]
        X_val, y_val = X_scaled[val_idx], y_np[val_idx]

        self.model = _RegressionNet(
            input_size=self.input_size,
            hidden_layers=self.hidden_layers,
            dropout=self.dropout,
            use_batch_norm=self.use_batch_norm,
        ).to(self.device)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=8, min_lr=1e-6
        )

        train_tensor = torch.tensor(X_train, device=self.device)
        train_target = torch.tensor(y_train, device=self.device)
        val_tensor = torch.tensor(X_val, device=self.device)
        val_target = torch.tensor(y_val, device=self.device)

        dataset = torch.utils.data.TensorDataset(train_tensor, train_target)
        generator = torch.Generator()
        generator.manual_seed(self.random_seed)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True, generator=generator
        )

        best_state = None
        best_val = float("inf")
        patience_counter = 0

        for epoch in range(self.max_epochs):
            self.model.train()
            epoch_loss = 0.0
            batches = 0
            for batch_x, batch_y in dataloader:
                optimizer.zero_grad()
                preds = self.model(batch_x)
                loss = criterion(preds, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                batches += 1

            self.model.eval()
            with torch.no_grad():
                val_preds = self.model(val_tensor)
                val_loss = criterion(val_preds, val_target).item()

            scheduler.step(val_loss)

            if val_loss < best_val - 1e-6:
                best_val = val_loss
                patience_counter = 0
                best_state = copy.deepcopy(self.model.state_dict())
            else:
                patience_counter += 1

            if self.verbose:
                avg_loss = epoch_loss / max(1, batches)
                print(f"[DNN] epoch={epoch+1} train_loss={avg_loss:.4f} val_loss={val_loss:.4f}")

            if patience_counter >= self.early_stopping_patience:
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self._is_fitted or self.model is None or self.scaler is None:
            raise RuntimeError("Model has not been fitted yet.")

        X_np = np.asarray(X, dtype=np.float32)
        X_scaled = self.scaler.transform(X_np).astype(np.float32)
        tensor_x = torch.tensor(X_scaled, device=self.device)
        self.model.eval()
        with torch.no_grad():
            preds = self.model(tensor_x).detach().cpu().numpy()
        return preds

    # -------- Feature importance (optional) --------
    def compute_feature_importances(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        feature_names: Optional[List[str]],
    ) -> Optional[List[dict]]:
        """
        DNN模型不支持传统的feature_importances_，返回None。
        """
        return None

