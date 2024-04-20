import torch
from transformers import DecisionTransformerModel, DecisionTransformerConfig
from transformers.models.decision_transformer import TrainableDT


def initialize_model(actiondim):
    config = DecisionTransformerConfig(
        vocab_size=actiondim,  # Size of the vocabulary
        hidden_size=768,  # Dimensionality of the encoder layers and the pooler layer
        num_hidden_layers=12,  # Number of hidden layers in the Transformer encoder
        num_attention_heads=12,  # Number of attention heads for each attention layer in the Transformer encoder
        intermediate_size=3072,  # Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder
        hidden_act="gelu",  # The non-linear activation function (function or string) in the encoder and pooler
        hidden_dropout_prob=0.1,  # The dropout probability for all fully connected layers in the embeddings, encoder, and pooler
        attention_probs_dropout_prob=0.1,  # The dropout ratio for the attention probabilities
        max_position_embeddings=512,  # The maximum sequence length that this model might ever be used with
        type_vocab_size=2,  # The vocabulary size of the `token_type_ids` passed into `BertModel`
        initializer_range=0.02,  # The stdev of the truncated_normal_initializer for initializing all weight matrices
    )

    # Create an instance of TrainableDT
    model = TrainableDT(config)
    return model