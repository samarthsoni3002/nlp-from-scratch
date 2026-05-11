import torch 

def create_data(file_path):

  with open(file_path,"r",encoding="utf-8") as f:
    file_text = f.read()

  file_data = file_text.split("\n")

  X = []
  y = [] 

  for i in file_data:
    x_y = i.split("\t")
    if(len(x_y)>1):
      X.append(x_y[0])
      y.append(x_y[1])


  return X,y


def translate_sentence(
    sentence,
    model,
    vocab_eng,
    vocab_fr,
    id_to_word_fr,
    preprocess_fn,
    device,
    max_len=30
):
    model.eval()

    sos_id = vocab_fr["<SOS>"]
    eos_id = vocab_fr["<EOS>"]
    unk_id_eng = vocab_eng["<UNK>"]

    tokens = tokens = preprocess_fn({"text": [sentence]})["tokens"][0]

  
    tokens = tokens + ["<EOS>"]


    input_ids = [vocab_eng.get(token, unk_id_eng) for token in tokens]

    
    input_tensor = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(device)

    input_lengths = torch.tensor([len(input_ids)], dtype=torch.long).to(device)

    generated_ids = []

    with torch.no_grad():
        
        encoder_outputs, hidden, cell = model.encoder(input_tensor, input_lengths)

      
        decoder_input = torch.tensor([[sos_id]], dtype=torch.long).to(device)

   
        for _ in range(max_len):
            predictions, hidden, cell = model.decoder(
                decoder_input,
                hidden,
                cell
            )

           
            next_token_logits = predictions[:, -1, :]

   
            next_token_id = next_token_logits.argmax(dim=-1)

            next_id = next_token_id.item()

            if next_id == eos_id:
                break

            generated_ids.append(next_id)

    
            decoder_input = next_token_id.unsqueeze(1)

    generated_tokens = [
        id_to_word_fr[idx]
        for idx in generated_ids
    ]

    return generated_tokens