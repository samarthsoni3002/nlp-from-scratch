import torch


def training_loop(
    model,
    train_dataloader,
    val_dataloader,
    num_epochs,
    loss_fn,
    optimizer,
):
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for inputs, targets in train_dataloader:
            optimizer.zero_grad()

            logits = model(inputs)
            loss = loss_fn(logits, targets)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_dataloader)
        train_losses.append(avg_train_loss)

        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for inputs, targets in val_dataloader:
                logits = model(inputs)
                loss = loss_fn(logits, targets)

                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_dataloader)
        val_losses.append(avg_val_loss)

        print(
            f"Epoch [{epoch + 1}/{num_epochs}] | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f}"
        )

    return train_losses, val_losses


def testing_loop(model, test_dataloader, loss_fn):
    model.eval()
    test_loss = 0.0

    with torch.no_grad():
        for inputs, targets in test_dataloader:
            logits = model(inputs)
            loss = loss_fn(logits, targets)

            test_loss += loss.item()

    avg_test_loss = test_loss / len(test_dataloader)
    print(f"Final Test Loss: {avg_test_loss:.4f}")

    return avg_test_loss