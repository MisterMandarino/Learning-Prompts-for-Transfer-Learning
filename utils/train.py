from tqdm.auto import tqdm
import torch
import torch.nn as nn
from pathlib import Path

def train_step(model, data_loader, optimizer, loss_fn, device):
    model.train()
    train_loss, train_acc = 0, 0

    for batch, (images, classes) in enumerate(data_loader):
        if images.shape[1] == 1:
            images = images.repeat(1,3,1,1)
        images, classes = images.to(device), classes.to(device)
        _ ,_, pred = model(images)
        loss = loss_fn(pred,classes)
        train_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        y_pred_class = torch.argmax(torch.softmax(pred, dim=1), dim=1)
        train_acc += (y_pred_class == classes).sum().item() / len(pred)

    train_loss = train_loss / len(data_loader)
    train_acc = train_acc / len(data_loader)
    return train_loss, train_acc

def test_step(model, data_loader, loss_fn, device):
    model.eval()

    test_loss, test_acc = 0, 0

    with torch.inference_mode():
        for batch, (images, classes) in enumerate(data_loader):
            if images.shape[1] == 1:
                images = images.repeat(1,3,1,1)
            images, classes = images.to(device), classes.to(device)
            _ ,_, pred = model(images)

            loss = loss_fn(pred, classes)
            test_loss += loss.item()

            pred_labels = pred.argmax(dim=1)
            test_acc += ((pred_labels == classes).sum().item() / len(pred_labels))

        test_loss = test_loss / len(data_loader)
        test_acc = test_acc / len(data_loader)
    return test_loss, test_acc

# 1. Take in various parameters required for training and test steps
def train(model, train_loader, test_loader, optimizer, loss_fn, device, epochs=5):
    # 2. Create empty results dictionary
    results = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}

    # 3. Loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model, train_loader, optimizer, loss_fn, device=device)
        test_loss, test_acc = test_step(model, test_loader, loss_fn, device=device)

        # 4. Print out what's happening
        print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_acc: {test_acc:.4f}"
        )

        # 5. Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

    # 6. Return the filled results at the end of the epochs
    return results

def save_model(model: torch.nn.Module,target_dir: str,model_name: str):
    """Saves a PyTorch model to a target directory.

    Args:
        model: A target PyTorch model to save.
        target_dir: A directory for saving the model to.
        model_name: A filename for the saved model. Should include
        either ".pth" or ".pt" as the file extension.

    Example usage:
        save_model(model=model_0,target_dir="models",model_name="05_going_modular_tingvgg_model.pth")
    """
    # Create target directory
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True,exist_ok=True)

    # Create model save path
    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
    model_save_path = target_dir_path / model_name

    # Save the model state_dict()
    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(),f=model_save_path)