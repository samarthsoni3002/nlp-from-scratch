import torch 

def weight_function(pair_count):

  alpha = 0.75
  x_max = 100

  if(pair_count < x_max):
    return (pair_count/x_max)**alpha
  else:
    return 1


def loss_fxn(score, function, pair_counts):
    weights = torch.tensor([function(x.item()) for x in pair_counts], dtype=torch.float32)
    log_counts = torch.log(pair_counts)
    loss = (score - log_counts) ** 2
    return (weights * loss).mean()