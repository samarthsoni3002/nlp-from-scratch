import torch
from utils import sample_negative_words

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

        for pairs in train_dataloader:
            optimizer.zero_grad()
            inputs, targets = pairs[0],pairs[1]

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
            for pairs in val_dataloader:
              
              inputs, targets = pairs[0], pairs[1]
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


def training_loop_negative_sampling_skipgram(model,
                                    train_dataloader,
                                    val_dataloader, 
                                    num_epochs, 
                                    loss_fn, 
                                    optimizer, 
                                    neg_samples, 
                                    noise_dist,
                                    counter, 
                                    vocab):

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        probab = noise_dist(counter,vocab)

        for pairs in train_dataloader:
            center_ids, pos_ids = pairs[0], pairs[1]

            neg_ids = torch.tensor(sample_negative_words(pos_ids, neg_samples, probab))

            pos_scores, neg_scores = model(center_ids, pos_ids, neg_ids)
            loss = loss_fn(pos_scores, neg_scores)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_dataloader)

        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for pairs in val_dataloader:
                center_ids, pos_ids = pairs[0], pairs[1]

                neg_ids = torch.tensor(sample_negative_words(pos_ids, neg_samples, probab))

                pos_scores, neg_scores = model(center_ids, pos_ids, neg_ids)
                loss = loss_fn(pos_scores, neg_scores)

                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_dataloader)

        print(f"Epoch [{epoch+1}/{num_epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")



def training_loop_negative_sampling_cbow(model, train_dataloader, val_dataloader, num_epochs, loss_fn, optimizer):

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for context_ids, target_ids, neg_ids in train_dataloader:

            pos_scores, neg_scores = model(context_ids, target_ids, neg_ids)
            loss = loss_fn(pos_scores, neg_scores)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_dataloader)

        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for context_ids, target_ids, neg_ids in val_dataloader:

                pos_scores, neg_scores = model(context_ids, target_ids, neg_ids)
                loss = loss_fn(pos_scores, neg_scores)

                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_dataloader)

        print(f"Epoch [{epoch+1}/{num_epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        

def testing_loop(model, test_dataloader, loss_fn):
    model.eval()
    test_loss = 0.0

    with torch.no_grad():
        for pairs in test_dataloader:
          
          inputs, targets = pairs[0], pairs[1]
          
          logits = model(inputs)
          loss = loss_fn(logits, targets)

          test_loss += loss.item()

    avg_test_loss = test_loss / len(test_dataloader)
    print(f"Final Test Loss: {avg_test_loss:.4f}")

    return avg_test_loss
  

