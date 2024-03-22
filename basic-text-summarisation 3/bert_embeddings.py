import torch


def get_sentence_embeddings(chunk, model, device):
    """
    Generates sentence embeddings for a given text chunk using a BERT model.

    Args:
    chunk (torch.Tensor): A tensor representing tokenized sentences.
    model (BertModel): Pretrained BERT model for generating embeddings.
    device (torch.device): The device (CPU or GPU) used for computations.

    Returns:
    list: A list containing the sentence embeddings.
    """
    embeddings = []

    with torch.no_grad():
        if chunk.dim() == 1:
            chunk = chunk.unsqueeze(0)

        # Limit the sequence length to 512 tokens for BERT
        if chunk.size(1) > 512:
            chunk = chunk[:, :512]

        chunk = chunk.to(device)

        # Feed the chunk through the BERT model
        outputs = model(chunk)

        # Extract and process the output to get sentence embeddings
        sentence_embedding = outputs[0].mean(1).squeeze().numpy()
        embeddings.append(sentence_embedding)

    return embeddings
