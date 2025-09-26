import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from model import SimpleCNN
from data_preprocessing import BearingDataset

def train_model(model, source_loader, target_loader, num_epochs=100, learning_rate=0.0001, coral_lambda=1.0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    classification_criterion = torch.nn.CrossEntropyLoss()

    target_iter = iter(target_loader)
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for source_images, source_labels in source_loader:
            try:
                target_images = next(target_iter)
            except StopIteration:
                target_iter = iter(target_loader)
                target_images = next(target_iter)

            source_images, source_labels = source_images.to(device), source_labels.to(device)
            target_images = target_images.to(device)

            optimizer.zero_grad()

            source_features = model.features(source_images)
            target_features = model.features(target_images)

            source_preds = model.classifier(source_features.view(source_features.size(0), -1))
            classification_loss = classification_criterion(source_preds, source_labels)
            coral_loss = compute_coral_loss(source_features.view(source_features.size(0), -1), target_features.view(target_features.size(0), -1))

            total_loss_batch = classification_loss + coral_lambda * coral_loss
            total_loss_batch.backward()
            optimizer.step()

            total_loss += total_loss_batch.item()

        print(f"Epoch {epoch + 1}/{num_epochs} | Loss: {total_loss:.4f}")

    return model

def compute_coral_loss(source_features, target_features):
    d = source_features.size(1)
    source_mean = torch.mean(source_features, 0, keepdim=True)
    source_centered = source_features - source_mean
    source_cov = (source_centered.t() @ source_centered) / (source_features.size(0) - 1)

    target_mean = torch.mean(target_features, 0, keepdim=True)
    target_centered = target_features - target_mean
    target_cov = (target_centered.t() @ target_centered) / (target_features.size(0) - 1)

    loss = torch.mean(torch.mul((source_cov - target_cov), (source_cov - target_cov)))
    return loss
