import warnings

warnings.filterwarnings("ignore")

import torch
import pickle
from tqdm import tqdm
from torch.utils.data import DataLoader
from model import ModelConfig, QuantFlow, QuantileLoss
from dataset import TimeSeriesDataset


def get_data(file="data.pkl"):

    with open(file, "rb") as f:
        data = pickle.load(f)

    X_train = data["X_trains"]
    y_train = data["y_trains"]
    X_test = data["X_tests"]
    y_test = data["y_tests"]
    n_vars = data["n_vars"]

    return X_train, y_train, X_test, y_test, n_vars


def get_model_and_data():
    X_train, y_train, X_test, y_test, n_vars = get_data()

    config = ModelConfig()
    config.n_vars = n_vars

    train_loader = DataLoader(
        TimeSeriesDataset(X_train, y_train, device=config.device),
        batch_size=4096,
        shuffle=False,
    )
    test_loader = DataLoader(
        TimeSeriesDataset(X_test, y_test, device=config.device),
        batch_size=4096,
        shuffle=False,
    )

    model = QuantFlow(config).to(config.device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model Parameters: {param_count:,}")

    return model, train_loader, test_loader, config, param_count


def train_model(
    checkpoint_path="quantflow_checkpoints.pth",
    onnx_path="quantflow.onnx",
    epochs=100,
    patience=10,
):
    model, train_loader, test_loader, config, param_count = get_model_and_data()
    criterion = QuantileLoss(config.quantiles)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    best_val_loss = float("inf")
    start_epoch = 0
    counter = 0

    train_losses = []
    val_losses = []

    # Training Loop
    for epoch in range(start_epoch, start_epoch + epochs):
        model.train()
        train_loss = 0
        for bx, by in tqdm(train_loader):
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for bx, by in test_loader:
                out = model(bx)
                loss = criterion(out, by)
                val_loss += loss.item()
        val_loss /= len(test_loader)
        val_losses.append(val_loss)
        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), checkpoint_path)
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping triggered.")
                break

    model.load_state_dict(torch.load(checkpoint_path))

    return model, train_losses, val_losses, config


def export_to_onnx(model, onnx_path="quantflow.onnx", config=None):
    dummy_input = torch.randn(1, config.seq_len, config.n_vars).to(config.device)
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )
    print(f"Model saved to {onnx_path}")


if __name__ == "__main__":
    model, train_losses, val_losses, config = train_model()
    export_to_onnx(model, config=config)
