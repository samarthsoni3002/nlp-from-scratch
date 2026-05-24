import torch 

def train(model, dataloader, optimizer, loss_fn, num_epochs, device):
    model.train()

    for epoch in range(num_epochs):
        total_loss = 0

        for batch in dataloader:
            input_ids, output_ids, ip_lengths, op_lengths = batch

            input_ids = input_ids.to(device)
            output_ids = output_ids.to(device)

            decoder_input = output_ids[:, :-1]
            labels = output_ids[:, 1:]

            outputs = model(input_ids, decoder_input)

            if isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs

            loss = loss_fn(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1)
            )

            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)

        print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {avg_loss:.4f}")