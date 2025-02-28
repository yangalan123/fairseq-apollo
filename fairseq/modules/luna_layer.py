# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Optional

import torch
import torch.nn as nn
from fairseq import utils
from fairseq.modules import LayerNorm, LunarMultiheadAttention, LunarCausalAttention
from fairseq.modules.quant_noise import quant_noise
from fairseq.modules.fairseq_dropout import FairseqDropout
from torch import Tensor


class LunaEncoderLayer(nn.Module):
    """Encoder layer block.

    In the original paper each operation (multi-head attention or FFN) is
    postprocessed with: `dropout -> add residual -> layernorm`. In the
    tensor2tensor code they suggest that learning is more robust when
    preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *args.encoder_normalize_before* to ``True``.

    Args:
        args (argparse.Namespace): parsed command-line arguments
    """

    def __init__(self, args, index):
        super().__init__()
        self.quant_noise = getattr(args, "quant_noise_pq", 0)
        self.quant_noise_block_size = getattr(args, "quant_noise_pq_block_size", 8)

        self.index = index
        self.embed_dim = args.encoder_embed_dim
        self.normalize_before = args.encoder_normalize_before

        self.dropout_module = FairseqDropout(args.dropout, module_name=self.__class__.__name__)

        self.self_attn = self.build_self_attention(self.embed_dim, args)
        self.self_attn_layer_norm = LayerNorm(self.embed_dim)
        self.self_atten_proj_layer_norm = LayerNorm(self.embed_dim)

        self.activation_fn = utils.get_activation_fn(activation=getattr(args, "activation_fn", "relu"))
        activation_dropout_p = getattr(args, "activation_dropout", 0)
        if activation_dropout_p == 0:
            # for backwards compatibility with models that use args.relu_dropout
            activation_dropout_p = getattr(args, "relu_dropout", 0)
        self.activation_dropout_module = FairseqDropout(float(activation_dropout_p), module_name=self.__class__.__name__)

        self.fc1 = self.build_fc1(self.embed_dim, args.encoder_ffn_embed_dim, self.quant_noise, self.quant_noise_block_size)
        self.fc2 = self.build_fc2(args.encoder_ffn_embed_dim, self.embed_dim, self.quant_noise, self.quant_noise_block_size)
        self.final_layer_norm = LayerNorm(self.embed_dim)

    def build_fc1(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(nn.Linear(input_dim, output_dim), p=q_noise, block_size=qn_block_size)

    def build_fc2(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(nn.Linear(input_dim, output_dim), p=q_noise, block_size=qn_block_size)

    def build_self_attention(self, embed_dim, args):
        return LunarMultiheadAttention(
            embed_dim,
            args.encoder_attention_heads,
            args.encoder_projected_attention_heads,
            dropout=args.attention_dropout,
            self_attention=True,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
        )

    def upgrade_state_dict_named(self, state_dict, name):
        """
        Rename layer norm states from `...layer_norms.0.weight` to
        `...self_attn_layer_norm.weight` and `...layer_norms.1.weight` to
        `...final_layer_norm.weight`
        """
        layer_norm_map = {"0": "self_attn_layer_norm", "1": "final_layer_norm"}
        for old, new in layer_norm_map.items():
            for m in ("weight", "bias"):
                k = "{}.layer_norms.{}.{}".format(name, old, m)
                if k in state_dict:
                    state_dict["{}.{}.{}".format(name, new, m)] = state_dict[k]
                    del state_dict[k]

    def forward(self, x, px, encoder_padding_mask, encoder_projected_padding_mask):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            px (Tensor): projected input to the layer of shape `(proj_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, seq_len)` where padding elements are indicated by ``1``.
            encoder_projected_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, proj_len)` where padding elements are indicated by ``1``.

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
            projected output of shape `(proj_len, batch, embed_dim)`
        """

        residual = x
        presidual = px
        # apply prev layer norm
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
            px = self.self_atten_proj_layer_norm(px)

        x, px, _ = self.self_attn(query=x, pquery=px, context=x,
                                  context_padding_mask=encoder_padding_mask,
                                  pcontext_padding_mask=encoder_projected_padding_mask)
        # apply dropout
        x = self.dropout_module(x)
        px = self.dropout_module(px)
        # residual
        x = residual + x
        px = presidual + px

        # apply post layer norm
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)
            px = self.self_atten_proj_layer_norm(px)

        #######################################################################
        # Feed-Forward Network
        residual = x
        # apply prev layer norm
        if self.normalize_before:
            x = self.final_layer_norm(x)

        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        x = self.fc2(x)
        # apply dropout
        x = self.dropout_module(x)
        # residual
        x = residual + x

        # apply post layer norm
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        return x, px


class LunaDecoderLayer(nn.Module):
    """Decoder layer block.

    In the original paper each operation (multi-head attention, encoder
    attention or FFN) is postprocessed with: `dropout -> add residual ->
    layernorm`. In the tensor2tensor code they suggest that learning is more
    robust when preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *args.decoder_normalize_before* to ``True``.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """

    def __init__(self, args, index):
        super().__init__()
        self.quant_noise = getattr(args, "quant_noise_pq", 0)
        self.quant_noise_block_size = getattr(args, "quant_noise_pq_block_size", 8)

        self.index = index
        self.normalize_before = args.decoder_normalize_before
        self.embed_dim = args.decoder_embed_dim

        self.dropout_module = FairseqDropout(args.dropout, module_name=self.__class__.__name__)

        self.self_attn = self.build_self_attention(self.embed_dim, args)

        # use layerNorm rather than FusedLayerNorm for exporting.
        # char_inputs can be used to determint this.
        # TODO  remove this once we update apex with the fix
        export = getattr(args, "char_inputs", False)
        self.self_attn_layer_norm = LayerNorm(self.embed_dim, export=export)

        self.encoder_attn = self.build_encoder_attention(self.embed_dim, args)
        self.encoder_attn_layer_norm = LayerNorm(self.embed_dim, export=export)
        self.encoder_atten_proj_layer_norm = LayerNorm(self.embed_dim, export=export)

        self.activation_fn = utils.get_activation_fn(activation=getattr(args, "activation_fn", "relu"))
        activation_dropout_p = getattr(args, "activation_dropout", 0)
        if activation_dropout_p == 0:
            # for backwards compatibility with models that use args.relu_dropout
            activation_dropout_p = getattr(args, "relu_dropout", 0)
        self.activation_dropout_module = FairseqDropout(float(activation_dropout_p), module_name=self.__class__.__name__)

        self.fc1 = self.build_fc1(self.embed_dim, args.decoder_ffn_embed_dim, self.quant_noise, self.quant_noise_block_size)
        self.fc2 = self.build_fc2(args.decoder_ffn_embed_dim, self.embed_dim, self.quant_noise, self.quant_noise_block_size)
        self.final_layer_norm = LayerNorm(self.embed_dim, export=export)

        self.need_attn = True
        self.onnx_trace = False

    def build_fc1(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(nn.Linear(input_dim, output_dim), q_noise, qn_block_size)

    def build_fc2(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(nn.Linear(input_dim, output_dim), q_noise, qn_block_size)

    def build_self_attention(self, embed_dim, args):
        return LunarCausalAttention(
            embed_dim,
            args.decoder_attention_heads,
            dropout=args.attention_dropout,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
        )

    def build_encoder_attention(self, embed_dim, args):
        return LunarMultiheadAttention(
            embed_dim,
            args.decoder_attention_heads,
            args.decoder_projected_attention_heads,
            dropout=args.attention_dropout,
            encoder_decoder_attention=True,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
        )

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def forward(
        self,
        x,
        px,
        encoder_out,
        encoder_padding_mask: Optional[torch.Tensor] = None,
        encoder_projected_padding_mask: Optional[torch.Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        self_attn_padding_mask: Optional[torch.Tensor] = None,
        need_attn: bool = False,
        need_head_weights: bool = False,
    ):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            px (Tensor): projected input to the layer of shape `(proj_len, batch, embed_dim)`
            encoder_out (Tensor): output from encoder of shape `(encoder_seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor, optional): binary
                ByteTensor of shape `(batch, src_len)` where padding
                elements are indicated by ``1``.
            encoder_projected_padding_mask (ByteTensor, optional): binary
                ByteTensor of shape `(batch, proj_len)` where padding
                elements are indicated by ``1``.
            need_attn (bool, optional): return attention weights
            need_head_weights (bool, optional): return attention weights
                for each head (default: return average over heads).

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
            projected output of shape `(proj_len, batch, embed_dim)`
        """
        if need_head_weights:
            need_attn = True

        static_px = px is None

        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)

        x, attn = self.self_attn(query=x, pquery=px,
                                 key_padding_mask=self_attn_padding_mask,
                                 pkey_padding_mask=encoder_projected_padding_mask,
                                 incremental_state=incremental_state,
                                 need_weights=False)

        x = self.dropout_module(x)
        x = residual + x
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        residual = x
        presidual = px
        if self.normalize_before:
            x = self.encoder_attn_layer_norm(x)
            px = self.encoder_atten_proj_layer_norm(px) if not static_px else None

        x, px, attn = self.encoder_attn(query=x, pquery=px, context=encoder_out,
                                        context_padding_mask=encoder_padding_mask,
                                        pcontext_padding_mask=encoder_projected_padding_mask,
                                        incremental_state=incremental_state,
                                        static_context=True,
                                        need_weights=need_attn or (not self.training and self.need_attn),
                                        need_head_weights=need_head_weights)
        # apply dropout
        x = self.dropout_module(x)
        px = self.dropout_module(px) if not static_px else None

        x = residual + x
        px = presidual + px if not static_px else None
        if not self.normalize_before:
            x = self.encoder_attn_layer_norm(x)
            px = self.encoder_atten_proj_layer_norm(px) if not static_px else None

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)

        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        x = residual + x
        if not self.normalize_before:
            x = self.final_layer_norm(x)

        if self.onnx_trace and incremental_state is not None:
            saved_state = self.self_attn._get_input_buffer(incremental_state)
            assert saved_state is not None
            if self_attn_padding_mask is not None:
                self_attn_state = [
                    saved_state["prev_key"],
                    saved_state["prev_value"],
                    saved_state["prev_key_padding_mask"],
                ]
            else:
                self_attn_state = [saved_state["prev_key"], saved_state["prev_value"]]
            return x, px, attn, self_attn_state
        return x, px, attn, None

    def make_generation_fast_(self, need_attn: bool = False, **kwargs):
        self.need_attn = need_attn
