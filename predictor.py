import torch
from nltk.tokenize import word_tokenize

def text_to_indices(sentence, vocab, max_len=60):
    unk_index = vocab.get('<unk>', 0)
    tokens = word_tokenize(sentence.lower())
    indices = [vocab.get(token, unk_index) for token in tokens]
    indices = indices[-max_len:]  
    padded = [0] * (max_len - len(indices)) + indices 
    return padded

def get_top_k_predictions(model, vocab, text, max_len=60, top_k=3):
    index_to_word = {idx: word for word, idx in vocab.items()}
    input_indices = text_to_indices(text, vocab, max_len)
    input_tensor = torch.tensor(input_indices, dtype=torch.long).unsqueeze(0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    input_tensor = input_tensor.to(device)

    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)

        k = min(top_k, probs.shape[1]) 
        top_indices = torch.topk(probs, k).indices[0].tolist()
        top_words = [index_to_word.get(idx, '<unk>') for idx in top_indices]

    return top_words
