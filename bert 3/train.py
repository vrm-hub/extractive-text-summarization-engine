import torch
import torch.optim as optim
import torch.nn as nn
from model import SummarizationModel


def train_model(train_loader, val_loader, learning_rate, num_epochs):
    """
    Trains the summarization model.

    Args:
    train_loader (DataLoader): DataLoader for the training dataset.
    val_loader (DataLoader): DataLoader for the validation dataset.
    learning_rate (float): Learning rate for the optimizer.
    num_epochs (int): Number of epochs to train the model.

    Returns:
    model: Trained summarization model.
    """
    # Setting up device, model, optimizer, and loss criterion
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SummarizationModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CosineEmbeddingLoss()

    best_val_loss = float('inf')

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        # Iterate over training batches
        for i, batch in enumerate(train_loader):
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            scores = batch['scores'].to(device)

            outputs = model(input_ids, attention_mask)
            outputs = outputs.view(-1, 1)  # Reshape outputs to [batch_size, 1]
            scores = scores.view(-1, 1)  # Reshape scores to [batch_size, 1]
            targets = torch.ones(outputs.size(0), device=device)

            loss = criterion(outputs, scores, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            print(f"Batch {i + 1}/{len(train_loader)} - Training Loss: {loss.item()}")

        # Validation loop
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for val_batch in val_loader:
                input_ids = val_batch['input_ids'].to(device)
                attention_mask = val_batch['attention_mask'].to(device)
                scores = val_batch['scores'].to(device)

                outputs = model(input_ids, attention_mask)
                outputs = outputs.view(-1, 1)
                scores = scores.view(-1, 1)
                targets = torch.ones(outputs.size(0), device=device)

                loss = criterion(outputs, scores, targets)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)

        # Update best model if current validation loss is lower
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_model.pt')

        print(
            f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {total_loss / len(train_loader)}, Val Loss: {avg_val_loss}")

    return model
