# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import utils
from fairseq.models import (
    FairseqEncoderModel,
    FairseqEncoder,
    register_model,
    register_model_architecture,
)
from fairseq.modules import (
    LayerNorm,
    SinusoidalPositionalEmbedding,
    TransformerSentenceEncoder,
    LunaSentenceEncoder
)
from fairseq.modules.transformer_sentence_encoder import init_bert_params
from fairseq.models.transformer import TransformerEncoder
from fairseq.models.luna import LunaEncoder

logger = logging.getLogger(__name__)


@register_model('transformer_lra')
class TransformerLRAModel(FairseqEncoderModel):
    """
    Class for training a transformer for LRA tasks.
    """
    def __init__(self, args, encoder, task):
        super().__init__(encoder)
        self.encoder = encoder
        self.args = args
        self.use_p = args.use_p
        self._max_positions = args.max_positions
        self.padding_idx = task.dictionary.pad()
        self.sentence_out_dim = args.sentence_class_num
        self.lm_output_learned_bias = None
        self.classifier = nn.ModuleList([])
        self.classifier.append(nn.Linear(args.classifier_in_dim, args.classifier_out_dim))
        self.classifier.extend([
            nn.Linear(args.classifier_out_dim, args.classifier_out_dim)
            for _ in range(args.classifier_layers - 1)
        ])
        # self.classifier = nn.Linear(args.classifier_in_dim, args.classifier_out_dim)
        self.classifier_activation = utils.get_activation_fn(args.classifier_activation_fn)
        self.sentence_projection_layer = nn.Linear(
            args.classifier_out_dim,
            self.sentence_out_dim,
            bias=False
        )
        self.sen_rep_type = getattr(args, "sen_rep_type", "first")

        # if specified then apply bert initialization on the model. We need
        # to explictly call this to make sure that the output embeddings
        # and projection layers are also correctly initialized
        if getattr(args, 'apply_bert_init', False):
            self.apply(init_bert_params)

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # Arguments related to dropout
        parser.add_argument('--dropout', type=float, metavar='D',
                            help='dropout probability')
        parser.add_argument('--attention-dropout', type=float,
                            metavar='D', help='dropout probability for'
                            ' attention weights')
        parser.add_argument('--act-dropout', type=float,
                            metavar='D', help='dropout probability after'
                            ' activation in FFN')

        # Arguments related to hidden states and self-attention
        parser.add_argument('--encoder-ffn-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension for FFN')
        parser.add_argument('--encoder-layers', type=int, metavar='N',
                            help='num encoder layers')
        parser.add_argument('--encoder-attention-heads', type=int, metavar='N',
                            help='num encoder attention heads')

        # Arguments related to input and output embeddings
        parser.add_argument('--encoder-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension')
        parser.add_argument('--share-encoder-input-output-embed',
                            action='store_true', help='share encoder input'
                            ' and output embeddings')
        parser.add_argument('--encoder-learned-pos', action='store_true',
                            help='use learned positional embeddings in the encoder')
        parser.add_argument('--no-token-positional-embeddings',
                            action='store_true',
                            help='if set, disables positional embeddings'
                            ' (outside self attention)')
        parser.add_argument('--num-segment', type=int, metavar='N',
                            help='num segment in the input')
        parser.add_argument('--max-positions', type=int,
                            help='number of positional embeddings to learn')

        # Arguments related to sentence level prediction
        parser.add_argument('--sentence-class-num', type=int, metavar='N',
                            help='number of classes for sentence task')
        parser.add_argument('--sent-loss', action='store_true', help='if set,'
                            ' calculate sentence level predictions')

        # Arguments related to parameter initialization
        parser.add_argument('--apply-bert-init', action='store_true',
                            help='use custom param initialization for BERT')
        
        parser.add_argument('--use-p', default=False, action='store_true',
                            help='use p for prediction')

        # misc params
        parser.add_argument('--activation-fn',
                            choices=utils.get_available_activation_fns(),
                            help='activation function to use')
        parser.add_argument('--classifier-activation-fn',
                            choices=utils.get_available_activation_fns(),
                            help='Which activation function to use for classifier layer.')
        parser.add_argument('--encoder-normalize-before', action='store_true',
                            help='apply layernorm before each encoder block')
        parser.add_argument('--encoder-layerdrop', type=float, metavar='D', default=0,
                            help='LayerDrop probability for encoder')
        parser.add_argument('--no-scale-embedding', action='store_true',
                            help='if True, dont scale embeddings')
        parser.add_argument('--quant-noise-pq', type=float, metavar='D', default=0,
                            help='iterative PQ quantization noise at training time')
        parser.add_argument('--quant-noise-pq-block-size', type=int, metavar='D', default=8,
                            help='block size of quantization noise at training time')
        parser.add_argument('--quant-noise-scalar', type=float, metavar='D', default=0,
                            help='scalar quantization noise and scalar quantization at training time')
        parser.add_argument(
            '--layer-type',
            choices=['transformer', 'luna']
        )
        parser.add_argument(
            '--sen-rep-type',
            choices=['first', 'mp']
        )
        parser.add_argument(
            '--encoder-projected-length', type=int, metavar='N',
            help='projected length of encoder as key'
        )
        parser.add_argument(
            '--encoder-projected-attention-heads', type=int, metavar='N',
            help='num encoder projected attention heads'
        )
        parser.add_argument(
            '--decoder-projected-attention-heads', type=int, metavar='N',
            help='num decoder projected attention heads'
        )

    def forward(self, sample):
        src_tokens = sample['net_input']['src_tokens']
        if self.use_p:
            src_tokens = src_tokens[:, 1:]
        sentence_rep = self.encoder(src_tokens)
        if not self.use_p:
            sentence_rep = sentence_rep[1]
        else:
            sentence_rep = sentence_rep[2].mean(dim=0)
        if 'net_input1' in sample:
            src1_tokens = sample['net_input1']
            sentence1_rep = self.encoder(src1_tokens)
            if not self.use_p:
                sentence1_rep = sentence1_rep[1]
            else:
                sentence1_rep = sentence1_rep[2].mean(dim=0)
            # sentence1_rep = encoder_out1.encoder_out[0,...]
            concat_rep = []
            concat_rep.append(sentence1_rep)
            concat_rep.append(sentence_rep)
            # concat_rep.append(sentence1_rep + sentence_rep)
            # concat_rep.append(sentence1_rep * sentence_rep)
            sentence_rep = torch.cat(concat_rep, dim=-1)
        for layer in self.classifier:
            sentence_rep = self.classifier_activation(layer(sentence_rep))
        if self.sentence_projection_layer:
            sentence_logits = self.sentence_projection_layer(sentence_rep)
        return {
            'encoder_out': sentence_logits
        }

    def max_positions(self):
        """Maximum output length supported by the encoder."""
        return self._max_positions

    # def upgrade_state_dict_named(self, state_dict, name):
        # if isinstance(
        #         self.encoder.embed_positions,
        #         SinusoidalPositionalEmbedding
        # ):
        #     state_dict[
        #         name + '.sentence_encoder.embed_positions._float_tensor'
        #     ] = torch.FloatTensor(1)
        # if not self.load_softmax:
        #     for k in list(state_dict.keys()):
        #         if (
        #             "embed_out.weight" in k or
        #             "sentence_projection_layer.weight" in k or
        #             "lm_output_learned_bias" in k
        #         ):
        #             del state_dict[k]
        # return state_dict
    
    @classmethod
    def build_embedding(cls, args, dictionary, embed_dim, path=None):
        num_embeddings = len(dictionary)
        padding_idx = dictionary.pad()

        emb = nn.Embedding(num_embeddings, embed_dim, padding_idx)
        # if provided, load from preloaded dictionaries
        if path:
            embed_dict = utils.parse_embedding(path)
            utils.load_embedding(embed_dict, dictionary, emb)
        return emb

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        # make sure all arguments are present in older models
        base_architecture(args)
        if not hasattr(args, 'max_positions'):
            args.max_positions = args.tokens_per_sample
        if not hasattr(args, 'max_source_positions'):
            args.max_source_positions = args.max_positions
        if not hasattr(args, 'decoder_embed_dim'):
            args.decoder_embed_dim = args.encoder_embed_dim
        embed_tokens = cls.build_embedding(args, task.dictionary, args.encoder_embed_dim)
        logger.info(args)
        encoder = TransformerLRAEncoder(args, task)
        return cls(args, encoder, task)


class TransformerLRAEncoder(FairseqEncoder):
    """RoBERTa encoder."""

    def __init__(self, args, task):
        super().__init__(task.dictionary)
        self.args = args

        if args.layer_type == 'transformer':
            self.encoder = TransformerSentenceEncoder(
                tie_layer_weights=getattr(args, 'tie_layer_weights', False),
                padding_idx=task.dictionary.pad_index,
                vocab_size=len(task.dictionary),
                num_encoder_layers=args.encoder_layers,
                embedding_dim=args.encoder_embed_dim,
                ffn_embedding_dim=args.encoder_ffn_embed_dim,
                num_attention_heads=args.encoder_attention_heads,
                dropout=args.dropout,
                attention_dropout=args.attention_dropout,
                activation_dropout=args.act_dropout,
                max_seq_len=args.max_positions,
                num_segments=0,
                use_position_embeddings=True,
                offset_positions_by_padding=True,
                encoder_normalize_before=True,
                apply_bert_init=True,
                activation_fn=args.activation_fn,
                learned_pos_embedding=True,
                normalize_before=False,
                sen_rep_type=getattr(args, 'sen_rep_type', 'cls')
            )
        else:
            self.encoder = LunaSentenceEncoder(
                tie_layer_weights=getattr(args, 'tie_layer_weights', False),
                projected_length=args.encoder_projected_length,
                padding_idx=task.dictionary.pad_index,
                vocab_size=len(task.dictionary),
                num_encoder_layers=args.encoder_layers,
                embedding_dim=args.encoder_embed_dim,
                ffn_embedding_dim=args.encoder_ffn_embed_dim,
                num_attention_heads=args.encoder_attention_heads,
                dropout=args.dropout,
                attention_dropout=args.attention_dropout,
                activation_dropout=args.act_dropout,
                max_seq_len=args.max_positions,
                use_position_embeddings=True,
                offset_positions_by_padding=True,
                layernorm_embedding=True,
                apply_bert_init=getattr(args, "apply_bert_init", False),
                activation_fn=args.activation_fn,
                learned_pos_embedding=True,
                embed_scale=None,
                sen_rep_type=getattr(args, 'sen_rep_type', 'cls'),
                no_scale_embedding=getattr(args, 'no_scale_embedding', True)
            )
    
    def forward(self, src_tokens):

        return self.encoder(src_tokens)

    # def forward(self, src_tokens, features_only=False, return_all_hiddens=False, masked_tokens=None, **unused):
    #     """
    #     Args:
    #         src_tokens (LongTensor): input tokens of shape `(batch, src_len)`
    #         features_only (bool, optional): skip LM head and just return
    #             features. If True, the output will be of shape
    #             `(batch, src_len, embed_dim)`.
    #         return_all_hiddens (bool, optional): also return all of the
    #             intermediate hidden states (default: False).

    #     Returns:
    #         tuple:
    #             - the LM output of shape `(batch, src_len, vocab)`
    #             - a dictionary of additional data, where 'inner_states'
    #               is a list of hidden states. Note that the hidden
    #               states have shape `(src_len, batch, vocab)`.
    #     """
    #     x, extra = self.extract_features(src_tokens, return_all_hiddens=return_all_hiddens)
    #     if not features_only:
    #         x = self.output_layer(x, masked_tokens=masked_tokens)
    #     return x, extra

    # def extract_features(self, src_tokens, return_all_hiddens=False, **unused):
    #     inner_states, _ = self.sentence_encoder(
    #         src_tokens,
    #         last_state_only=not return_all_hiddens,
    #     )
    #     features = inner_states[-1].transpose(0, 1)  # T x B x C -> B x T x C
    #     return features, {'inner_states': inner_states if return_all_hiddens else None}

    # def output_layer(self, features, masked_tokens=None, **unused):
    #     return self.lm_head(features, masked_tokens)

    # def max_positions(self):
    #     """Maximum output length supported by the encoder."""
    #     return self.args.max_positions

@register_model_architecture('transformer_lra', 'transformer_lra')
def base_architecture(args):
    args.dropout = getattr(args, 'dropout', 0.1)
    args.attention_dropout = getattr(args, 'attention_dropout', 0.1)
    args.act_dropout = getattr(args, 'act_dropout', 0.0)

    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 2048)
    args.encoder_layers = getattr(args, 'encoder_layers', 6)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 8)

    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 512)
    args.share_encoder_input_output_embed = getattr(args, 'share_encoder_input_output_embed', False)
    args.encoder_learned_pos = getattr(args, 'encoder_learned_pos', True)
    args.no_token_positional_embeddings = getattr(args, 'no_token_positional_embeddings', False)
    args.num_segment = getattr(args, 'num_segment', 0)
    args.classifier_out_dim = getattr(args, 'classifier_out_dim', 2048)

    args.sentence_class_num = getattr(args, 'sentence_class_num', 2)
    args.sent_loss = getattr(args, 'sent_loss', True)

    args.apply_bert_init = getattr(args, 'apply_bert_init', False)

    args.activation_fn = getattr(args, 'activation_fn', 'gelu')
    args.classifier_layers = getattr(args, 'classifier_layers', 1)
    args.classifier_activation_fn = getattr(args, 'classifier_activation_fn', 'relu')
    args.encoder_normalize_before = getattr(args, 'encoder_normalize_before', True)
    args.layer_type = getattr(args, 'layer_type', 'transformer')
    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.classifier_in_dim = getattr(args, "classifier_in_dim", args.encoder_embed_dim)


@register_model_architecture('transformer_lra', 'transformer_lra_listop')
def transformer_lra_listop(args):
    args.sentence_class_num = getattr(args, 'sentence_class_num', 10)
    args.max_positions = getattr(args, 'max_positions', 2002)
    args.tie_layer_weights = getattr(args, 'tie_layer_weights', True)
    base_architecture(args)

@register_model_architecture('transformer_lra', 'transformer_lra_usep_listop')
def transformer_lra_listop(args):
    args.use_p = getattr(args, 'use_p', True)
    args.sentence_class_num = getattr(args, 'sentence_class_num', 10)
    args.max_positions = getattr(args, 'max_positions', 2002)
    args.tie_layer_weights = getattr(args, 'tie_layer_weights', True)
    base_architecture(args)

@register_model_architecture('transformer_lra', 'luna_lra_listop')
def luna_lra_listop(args):
    args.sentence_class_num = getattr(args, 'sentence_class_num', 10)
    args.max_positions = getattr(args, 'max_positions', 2001)
    args.tie_layer_weights = getattr(args, 'tie_layer_weights', True)
    args.layer_type = getattr(args, 'layer_type', 'luna')
    base_architecture(args)

@register_model_architecture('transformer_lra', 'transformer_lra_imdb')
def transformer_lra_imdb_architecture(args):
    args.max_positions = getattr(args, 'max_positions', 1002)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 1024)
    args.encoder_layers = getattr(args, 'encoder_layers', 4)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 256)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 4)
    args.classifier_layers = getattr(args, 'classifier_layers', 1)
    args.classifier_out_dim = getattr(args, 'classifier_out_dim', 1024)
    base_architecture(args)

@register_model_architecture('transformer_lra', 'transformer_lra_imdb_eff')
def transformer_lra_imdb_eff_architecture(args):
    args.max_positions = getattr(args, 'max_positions', 1000)
    transformer_lra_imdb_architecture(args)

@register_model_architecture('transformer_lra', 'luna_lra_imdb')
def luna_lra_imdb_architecture(args):
    args.layer_type = getattr(args, 'layer_type', 'luna')
    transformer_lra_imdb_architecture(args)

@register_model_architecture('transformer_lra', 'luna_lra_imdb_eff')
def luna_lra_imdb_architecture(args):
    args.max_positions = getattr(args, 'max_positions', 2000)
    args.layer_type = getattr(args, 'layer_type', 'luna')
    transformer_lra_imdb_architecture(args)

@register_model_architecture('transformer_lra', 'transformer_lra_aan')
def transformer_lra_aan_architecture(args):
    args.apply_bert_init = getattr(args, 'apply_bert_init', True)
    args.max_positions = getattr(args, 'max_positions', 4002)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 512)
    args.encoder_layers = getattr(args, 'encoder_layers', 4)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 128)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 4)
    args.classifier_layers = getattr(args, 'classifier_layers', 1)
    args.classifier_out_dim = getattr(args, 'classifier_out_dim', 512)
    args.classifier_in_dim = getattr(args, 'classifier_in_dim', args.encoder_embed_dim * 2)
    base_architecture(args)

@register_model_architecture('transformer_lra', 'luna_lra_aan')
def luna_lra_aan_architecture(args):
    args.layer_type = getattr(args, 'layer_type', 'luna')
    transformer_lra_aan_architecture(args)

@register_model_architecture('transformer_lra', 'transformer_lra_cifar10')
def transformer_lra_cifar10(args):
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 128)
    args.encoder_layers = getattr(args, 'encoder_layers', 1)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 64)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 8)
    args.classifier_layers = getattr(args, 'classifier_layers', 1)
    args.classifier_out_dim = getattr(args, 'classifier_out_dim', 128)
    args.sentence_class_num = getattr(args, 'sentence_class_num', 10)
    args.max_positions = getattr(args, 'max_positions', 1024)
    # args.dropout = getattr(args, 'dropout', 0.3)
    # args.attention_dropout = getattr(args, 'attention_dropout', 0.1)
    # args.tie_layer_weights = getattr(args, 'tie_layer_weights', True)
    base_architecture(args)

@register_model_architecture('transformer_lra', 'luna_lra_cifar10')
def luna_lra_cifar10(args):
    args.layer_type = getattr(args, 'layer_type', 'luna')
    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    transformer_lra_cifar10(args)

@register_model_architecture('transformer_lra', 'transformer_lra_pf32')
def transformer_lra_pf32(args):
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 128)
    args.encoder_layers = getattr(args, 'encoder_layers', 1)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 128)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 4)
    args.classifier_layers = getattr(args, 'classifier_layers', 1)
    args.classifier_out_dim = getattr(args, 'classifier_out_dim', 256)
    args.sentence_class_num = getattr(args, 'sentence_class_num', 2)
    args.max_positions = getattr(args, 'max_positions', 1026)
    # args.tie_layer_weights = getattr(args, 'tie_layer_weights', True)
    # args.dropout = getattr(args, 'dropout', 0.2)
    # args.attention_dropout = getattr(args, 'attention_dropout', 0.2)
    args.sen_rep_type = getattr(args, 'sen_rep_type', 'mp')
    base_architecture(args)

@register_model_architecture('transformer_lra', 'luna_lra_pf32')
def luna_lra_pf32(args):
    # args.dropout = getattr(args, 'dropout', 0.2)
    # args.attention_dropout = getattr(args, 'attention_dropout', 0.2)
    args.layer_type = getattr(args, 'layer_type', 'luna')
    transformer_lra_pf32(args)

@register_model_architecture('transformer_lra', 'luna_lra_pf128')
def luna_lra_pf32(args):
    args.max_positions = getattr(args, 'max_positions', 128*128+2)
    # args.dropout = getattr(args, 'dropout', 0.2)
    # args.attention_dropout = getattr(args, 'attention_dropout', 0.2)
    args.layer_type = getattr(args, 'layer_type', 'luna')
    transformer_lra_pf32(args)