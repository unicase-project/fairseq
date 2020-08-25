# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Unicase Model
"""
from typing import Optional, Tuple

from fairseq.models import register_model, register_model_architecture

from .hub_interface import RobertaHubInterface
from .model import RobertaModel, base_architecture, RobertaEncoder

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
RoBERTa: A Robustly Optimized BERT Pretraining Approach.
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import utils
from fairseq.models import (
    FairseqEncoder,
    FairseqEncoderModel,
    register_model,
    register_model_architecture,
)
from fairseq.modules import (
    LayerNorm,
    TransformerSentenceEncoder, TransformerSentenceEncoderLayer,
)
from fairseq.modules.transformer_sentence_encoder import init_bert_params
from fairseq.modules.quant_noise import quant_noise as apply_quant_noise_

from .hub_interface import RobertaHubInterface

logger = logging.getLogger(__name__)


class UnicaseSentenceEncoder(TransformerSentenceEncoder):
    """

    """

    def __init__(
            self,
            padding_idx: int,
            vocab_size: int,
            num_encoder_layers: int = 6,
            embedding_dim: int = 768,
            ffn_embedding_dim: int = 3072,
            num_attention_heads: int = 8,
            dropout: float = 0.1,
            attention_dropout: float = 0.1,
            activation_dropout: float = 0.1,
            layerdrop: float = 0.0,
            max_seq_len: int = 256,
            num_segments: int = 2,
            use_position_embeddings: bool = True,
            offset_positions_by_padding: bool = True,
            encoder_normalize_before: bool = False,
            apply_bert_init: bool = False,
            activation_fn: str = "relu",
            learned_pos_embedding: bool = True,
            embed_scale: float = None,
            freeze_embeddings: bool = False,
            n_trans_layers_to_freeze: int = 0,
            export: bool = False,
            traceable: bool = False,
            q_noise: float = 0.0,
            qn_block_size: int = 8,
            dict_cased_words: int = 0,
            dict_non_cased_words: int = 0,
            dict_nspecial: int = 0,
    ) -> None:

        super().__init__(padding_idx, vocab_size, num_encoder_layers, embedding_dim,
                         ffn_embedding_dim, num_attention_heads, dropout, attention_dropout,
                         activation_dropout, layerdrop, max_seq_len, num_segments,
                         use_position_embeddings, offset_positions_by_padding,
                         encoder_normalize_before, apply_bert_init, activation_fn,
                         learned_pos_embedding, embed_scale, freeze_embeddings,
                         n_trans_layers_to_freeze, export, traceable, q_noise, qn_block_size)

        assert dict_nspecial != 0 and dict_non_cased_words != 0 and dict_cased_words != 0, \
            "All dictionary options need to be passed"

        self.dict_non_cased_words = dict_non_cased_words
        self.dict_cased_words = dict_cased_words
        self.dict_nspecial = dict_nspecial
        self.base_token_size = self.vocab_size - 2 * (self.dict_cased_words // 3)
        self.embed_tokens = self.build_embedding(
            self.base_token_size, self.embedding_dim, self.padding_idx
        )

        self.embed_case = self.build_embedding(
            4, self.embedding_dim, None
        )

        # Apply initialization of model params after building the model
        if self.apply_bert_init:
            self.apply(init_bert_params)

    @torch.no_grad()
    def get_unicase_ids(self, token_ids):
        all_non_cased = self.dict_nspecial + self.dict_non_cased_words
        word_offset = (token_ids - all_non_cased).clamp(0)
        reminder = word_offset.remainder(3)
        floor_div = word_offset.floor_divide(3)
        token_base_ids = token_ids - reminder - 2 * floor_div
        case_ids = (reminder + 1) * (token_ids.ge(all_non_cased) *
                                    token_ids.lt(all_non_cased + self.dict_cased_words))

        return token_base_ids, case_ids

    def forward(
            self,
            tokens: torch.Tensor,
            segment_labels: torch.Tensor = None,
            last_state_only: bool = False,
            positions: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        # compute padding mask. This is needed for multi-head attention
        padding_mask = tokens.eq(self.padding_idx)
        if not self.traceable and not self.tpu and not padding_mask.any():
            padding_mask = None

        token_base_ids, case_ids = self.get_unicase_ids(tokens)

        x = self.embed_tokens(token_base_ids)
        x += self.embed_case(case_ids)

        if self.embed_scale is not None:
            x *= self.embed_scale

        if self.embed_positions is not None:
            x += self.embed_positions(tokens, positions=positions)

        if self.segment_embeddings is not None and segment_labels is not None:
            x += self.segment_embeddings(segment_labels)

        if self.quant_noise is not None:
            x = self.quant_noise(x)

        if self.emb_layer_norm is not None:
            x = self.emb_layer_norm(x)

        x = self.dropout_module(x)

        # account for padding while computing the representation
        if padding_mask is not None:
            x *= 1 - padding_mask.unsqueeze(-1).type_as(x)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        inner_states = []
        if not last_state_only:
            inner_states.append(x)

        for layer in self.layers:
            x, _ = layer(x, self_attn_padding_mask=padding_mask)
            if not last_state_only:
                inner_states.append(x)

        sentence_rep = x[0, :, :]

        if last_state_only:
            inner_states = [x]

        if self.traceable:
            return torch.stack(inner_states), sentence_rep
        else:
            return inner_states, sentence_rep


@register_model('unicase')
class UnicaseModel(RobertaModel):
    def __init__(self, args, encoder):
        super().__init__(args, encoder)

    @staticmethod
    def add_args(parser):
        RobertaModel.add_args(parser)
        """Add model-specific arguments to the parser."""
        parser.add_argument('--dict_non_cased_words', type=int, help='number of non-cased words'
                                                                     'at the begining of dict')
        parser.add_argument('--dict_cased_words', type=int, help='number of cased words which'
                                                                 'occur in triplets in dict')

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present
        base_architecture(args)

        if not hasattr(args, 'max_positions'):
            args.max_positions = args.tokens_per_sample

        dictionary = task.source_dictionary
        encoder = UnicaseEncoder(args, dictionary)
        return cls(args, encoder)

    def get_normalized_probs(self, net_output, log_probs, sample=None):
        """Get normalized probabilities (or log probs) from a net's output."""
        logits = net_output[0].float()
        if log_probs:
            return F.log_softmax(logits, dim=-1)
        else:
            return F.softmax(logits, dim=-1)

    @classmethod
    def from_pretrained(cls, model_name_or_path, checkpoint_file='model.pt', data_name_or_path='.',
                        bpe='sentencepiece', **kwargs):
        from fairseq import hub_utils
        x = hub_utils.from_pretrained(
            model_name_or_path,
            checkpoint_file,
            data_name_or_path,
            archive_map=cls.hub_models(),
            bpe=bpe,
            load_checkpoint_heads=True,
            **kwargs,
        )
        return RobertaHubInterface(x['args'], x['task'], x['models'][0])


class RobertaLMHead(nn.Module):
    """Head for masked language modeling."""

    def __init__(self, embed_dim, output_dim, activation_fn, weight=None):
        super().__init__()
        self.dense = nn.Linear(embed_dim, embed_dim)
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.layer_norm = LayerNorm(embed_dim)

        if weight is None:
            weight = nn.Linear(embed_dim, output_dim, bias=False).weight
        self.weight = weight
        self.bias = nn.Parameter(torch.zeros(output_dim))

    def forward(self, features, masked_tokens=None, **kwargs):
        # Only project the masked tokens while training,
        # saves both memory and computation
        if masked_tokens is not None:
            features = features[masked_tokens, :]

        x = self.dense(features)
        x = self.activation_fn(x)
        x = self.layer_norm(x)
        # project back to size of vocabulary with bias
        x = F.linear(x, self.weight) + self.bias
        return x


class UnicaseEncoder(FairseqEncoder):
    """RoBERTa encoder."""

    def __init__(self, args, dictionary):
        super().__init__(dictionary)
        self.args = args

        if args.encoder_layers_to_keep:
            args.encoder_layers = len(args.encoder_layers_to_keep.split(","))

        assert args.dict_cased_words % 3 == 0, "Cased words need to occur in triplets in dictionary"
        if dictionary[-1] == '<mask>':
            assert (args.dict_cased_words + args.dict_non_cased_words +
                    dictionary.nspecial) + 1 == len(dictionary)
        else:
            assert (args.dict_cased_words + args.dict_non_cased_words +
                    dictionary.nspecial) == len(dictionary)

        self.sentence_encoder = UnicaseSentenceEncoder(
            padding_idx=dictionary.pad(),
            vocab_size=len(dictionary),
            num_encoder_layers=args.encoder_layers,
            embedding_dim=args.encoder_embed_dim,
            ffn_embedding_dim=args.encoder_ffn_embed_dim,
            num_attention_heads=args.encoder_attention_heads,
            dropout=args.dropout,
            attention_dropout=args.attention_dropout,
            activation_dropout=args.activation_dropout,
            layerdrop=args.encoder_layerdrop,
            max_seq_len=args.max_positions,
            num_segments=0,
            encoder_normalize_before=True,
            apply_bert_init=True,
            activation_fn=args.activation_fn,
            q_noise=args.quant_noise_pq,
            qn_block_size=args.quant_noise_pq_block_size,
            dict_cased_words=args.dict_cased_words,
            dict_non_cased_words=args.dict_non_cased_words,
            dict_nspecial=dictionary.nspecial,
        )
        args.untie_weights_roberta = getattr(args, 'untie_weights_roberta', False)

        self.lm_head = RobertaLMHead(
            embed_dim=args.encoder_embed_dim,
            output_dim=len(dictionary) - 2 * (args.dict_cased_words // 3),
            activation_fn=args.activation_fn,
            weight=self.sentence_encoder.embed_tokens.weight if not args.untie_weights_roberta else None,
        )

        self.case_head = RobertaLMHead(
            embed_dim=args.encoder_embed_dim,
            output_dim=4,
            activation_fn=args.activation_fn,
            weight=self.sentence_encoder.embed_case.weight if not args.untie_weights_roberta else None,
        )

    def forward(self, src_tokens, features_only=False, return_all_hiddens=False, masked_tokens=None,
                **unused):
        """
        Args:
            src_tokens (LongTensor): input tokens of shape `(batch, src_len)`
            features_only (bool, optional): skip LM head and just return
                features. If True, the output will be of shape
                `(batch, src_len, embed_dim)`.
            return_all_hiddens (bool, optional): also return all of the
                intermediate hidden states (default: False).

        Returns:
            tuple:
                - the LM output of shape `(batch, src_len, vocab)`
                - a dictionary of additional data, where 'inner_states'
                  is a list of hidden states. Note that the hidden
                  states have shape `(src_len, batch, vocab)`.
        """
        x, extra = self.extract_features(src_tokens, return_all_hiddens=return_all_hiddens)
        if not features_only:
            extra["case_output"] = self.case_layer(x, masked_tokens=masked_tokens)
            x = self.output_layer(x, masked_tokens=masked_tokens)
        return x, extra

    def extract_features(self, src_tokens, return_all_hiddens=False, **unused):
        inner_states, _ = self.sentence_encoder(
            src_tokens,
            last_state_only=not return_all_hiddens,
        )
        features = inner_states[-1].transpose(0, 1)  # T x B x C -> B x T x C
        return features, {'inner_states': inner_states if return_all_hiddens else None}

    def output_layer(self, features, masked_tokens=None, **unused):
        return self.lm_head(features, masked_tokens)

    def case_layer(self, features, masked_tokens=None, **unused):
        return self.case_head(features, masked_tokens)

    def max_positions(self):
        """Maximum output length supported by the encoder."""
        return self.args.max_positions


@register_model_architecture('unicase', 'unicase_base')
def unicase_base_architecture(args):
    base_architecture(args)
