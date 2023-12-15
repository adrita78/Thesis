projection_dim = 128  
joint_model = JointModel(gnn_model, protbert_model, projection_dim)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(joint_model.parameters(), lr=0.001)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
joint_model.to(device)

def train_binary_classification(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    all_labels = []
    all_predictions = []

    for batch in tqdm(loader, desc="Training"):

        gnn_input, protbert_input, labels = batch
        gnn_input, protbert_input, labels = gnn_input.to(device), protbert_input.to(device), labels.to(device)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        logits = model(gnn_input, protbert_input)
        loss = criterion(logits, labels.unsqueeze(1).float())

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        predictions = (torch.sigmoid(logits) > 0.5).int()
        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predictions.cpu().numpy())
    accuracy = accuracy_score(all_labels, all_predictions)

    return total_loss / len(loader), accuracy

# Example usage
num_epochs = 10

for epoch in range(num_epochs):
    train_loss, train_accuracy = train_binary_classification(joint_model, dataloader, criterion, optimizer, device)
    print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {train_loss}, Accuracy: {train_accuracy}")
