import torch

def translate_sentence(
    sentence,
    model,
    vocab_eng,
    vocab_fr,
    id_to_word_fr,
    preprocess_fn,
    device,
    max_len=30,
):
    model.eval()
    
    sos_id = vocab_fr["<SOS>"]
    eos_id = vocab_fr["<EOS>"]
    unk_id_eng = vocab_eng["<UNK>"]
    
    tokens = preprocess_fn([sentence])["tokens"][0]
    tokens = tokens + ["<EOS>"]
    input_ids = [vocab_eng.get(token, unk_id_eng) for token in tokens]
    

    input_tensor = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(device)
    

    src_mask = model.create_src_mask(input_tensor)
    with torch.no_grad():
        encoder_outputs, enc_attn = model.encoder(input_tensor, src_mask)
        

        generated_ids = []
        decoder_input = torch.tensor([[sos_id]], dtype=torch.long).to(device)
        

        for _ in range(max_len):
      
            decoder_outputs, self_attn, cross_attn = model.decoder(
                decoder_input, 
                encoder_outputs, 
                src_mask
            )
            
      
            logits = model.fc_out(decoder_outputs)
            

            next_token_id = logits[:, -1, :].argmax(dim=-1)
            next_id = next_token_id.item()
            
            if next_id == eos_id:
                break
                
            generated_ids.append(next_id)

         
            decoder_input = torch.cat([decoder_input, next_token_id.unsqueeze(0)], dim=1)
  
    generated_tokens = [id_to_word_fr[idx] for idx in generated_ids]
    return generated_tokens