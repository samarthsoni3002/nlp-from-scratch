import torch 

def train(model, dataloader, optimizer, criterion, device):
    model.train()

    total_loss = 0

    for input_ids, output_ids, input_lengths, output_lengths in dataloader:
        input_ids = input_ids.to(device)
        output_ids = output_ids.to(device)
        input_lengths = input_lengths.to(device)


        predictions, attention_weights = model(input_ids, output_ids, input_lengths)

        decoder_target = output_ids[:, 1:]


        loss = criterion(
            predictions.reshape(-1, predictions.shape[-1]),
            decoder_target.reshape(-1)
        )

        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)

    return avg_loss