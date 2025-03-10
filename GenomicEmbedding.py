import torch
import torch.nn as nn
import torch.utils.checkpoint
from packaging import version
from transformers.models.bert.modeling_bert import BertEmbeddings


class GenomicEmbedding(BertEmbeddings):
    """Custom Bert Embeddings that modifies the position and word embeddings based on configuration."""

    def __init__(self, config):
        super().__init__(config)

        # Handling custom word embedding probability
        self.word_embeddings = nn.Embedding(
            config.vocab_size,
            round(config.hidden_size * config.word_embedding_prob),
            padding_idx=config.pad_token_id
        )
        self.pattern = config.pattern
        # Custom position embeddings based on pattern and other parameters in the configuration
        if config.word_embedding_prob != 1 and config.pattern == 2:
            self.position_embeddings_chrome = nn.Embedding(config.max_chrom, config.hidden_size - round(
                config.hidden_size * config.word_embedding_prob))
            self.position_embeddings_snp = nn.Embedding(config.max_each_chrom_snp, config.hidden_size - round(
                config.hidden_size * config.word_embedding_prob))

        if config.word_embedding_prob != 1 and config.pattern == 1:
            self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size - round(
                config.hidden_size * config.word_embedding_prob))

        # Additional linear layer and activation function
        self.Linear = nn.Linear(config.hidden_size, config.hidden_size)
        self.tanh = nn.Tanh()

        # LayerNorm and Dropout
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # Register position_ids and token_type_ids (as shown in the parent class)
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))

        # Handle token_type_ids (if needed)
        if version.parse(torch.__version__) > version.parse("1.6.0"):
            self.register_buffer(
                "token_type_ids",
                torch.zeros(self.position_ids.size(), dtype=torch.long),
                persistent=False,
            )

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None,
                past_key_values_length=0):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length: seq_length + past_key_values_length]

        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                token_type_ids = buffered_token_type_ids.expand(input_shape[0], seq_length)
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        embeddings = inputs_embeds

        # Add custom position embeddings
        if self.position_embedding_type == "absolute":
            if self.pattern == 2:
                chrome_tensor = torch.tensor(torch.load("./chrom_names.ped")).cuda()
                snp_tensor = torch.tensor(torch.load("./snp_sort.ped")).cuda()
                position_embeddings = (self.position_embeddings_chrome(chrome_tensor) + self.position_embeddings_snp(
                    snp_tensor)).expand(embeddings.shape[0], -1, -1)

            if self.pattern == 1:
                position_embeddings = (self.position_embeddings(position_ids)).expand(embeddings.shape[0], -1, -1)
            embeddings = torch.cat([position_embeddings, embeddings], dim=2)  # Concatenate position and word embeddings

            embeddings = self.Linear(embeddings)  # Linear transformation
            embeddings = self.tanh(embeddings)  # Activation function

        # Apply LayerNorm and Dropout
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings
