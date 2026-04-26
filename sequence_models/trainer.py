import torch 

def training_loop(model, train_dataloader, val_dataloader, num_epochs, loss_fn, optimizer):
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for input_ids, labels, lengths in train_dataloader:
            
            logits, hidden_states = model(input_ids, lengths)

            loss = loss_fn(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_dataloader)

        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for input_ids, labels, lengths in val_dataloader:
                
                logits, hidden_states = model(input_ids, lengths)
                loss = loss_fn(logits, labels)

                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_dataloader)

        print(f"Epoch [{epoch+1}/{num_epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        
def testing_loop(model, test_dataloader, loss_fn):
    model.eval()
    test_loss = 0.0

    with torch.no_grad():
        for input_ids, labels, lengths in test_dataloader:
            
            logits, hidden_states = model(input_ids, lengths)
            loss = loss_fn(logits, labels)

            test_loss += loss.item()

    avg_test_loss = test_loss / len(test_dataloader)
    print(f"Final Test Loss: {avg_test_loss:.4f}")

    return avg_test_loss