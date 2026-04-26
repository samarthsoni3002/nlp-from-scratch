def expand_contractions(text, contractions):

    for contraction, expanded in contractions.items():
        text = text.replace(contraction, expanded)

    return text