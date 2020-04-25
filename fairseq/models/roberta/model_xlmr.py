# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Unsupervised Cross-lingual Representation Learning at Scale
"""

from fairseq.models import register_model, register_model_architecture

from .hub_interface import RobertaHubInterface
from .model import RobertaModel, base_architecture


@register_model('xlmr')
class XLMRModel(RobertaModel):

    @classmethod
    def hub_models(cls):
        return {
            'xlmr.base': 'http://dl.fbaipublicfiles.com/fairseq/models/xlmr.base.tar.gz',
            'xlmr.large': 'http://dl.fbaipublicfiles.com/fairseq/models/xlmr.large.tar.gz',
        }

    @classmethod
    def from_pretrained(cls, model_name_or_path, checkpoint_file='model.pt', data_name_or_path='.', bpe='sentencepiece', **kwargs):
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


@register_model_architecture('xlmr', 'xlmr_base')
def xlm_architecture(args):
    base_architecture(args)


@register_model_architecture('xlmr', 'xlmr_large')
def xlm_architecture(args):
    args.encoder_layers = getattr(args, 'encoder_layers', 24)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 1024)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 4096)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 16)
    base_architecture(args)
