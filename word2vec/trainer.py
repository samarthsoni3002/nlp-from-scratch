import torch 


def training_loop(model,train_dataloader,val_dataloader,num_epochs,loss_fn,optimizer):

  for epoch in range(num_epochs):

    train_loss = 0

    for pairs in train_dataloader:

      logits = model(pairs[0])
      loss = loss_fn(logits,pairs[1])
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      train_loss += loss.item()
    avg_train_loss = train_loss / len(train_dataloader)


    model.eval()
    val_loss = 0

    for pairs in val_dataloader:
      inputs, targets = pairs[0], pairs[1]
      logits = model(inputs)
      loss = loss_fn(logits,targets)
      val_loss += loss.item()

    avg_val_loss = val_loss/ len(val_dataloader)

    print(f"Epoch [{epoch+1}/{num_epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")


def testing_loop(model, test_loader, loss_fn):

    model.eval()
    test_loss = 0

    with torch.no_grad():
        for batch in test_loader:

            inputs, targets = batch[0], batch[1]

            logits = model(inputs)
            loss = loss_fn(logits, targets)
            test_loss += loss.item()

    avg_test_loss = test_loss / len(test_loader)
    print(f"Final Test Loss: {avg_test_loss:.4f}")

    return avg_test_loss