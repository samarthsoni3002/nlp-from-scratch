def expand_contractions(text, contractions):

  for contractions, expanded in contractions.items():
    text = text.replace(contractions, expanded)

  return text