import os
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.cuda.amp import autocast

from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import Conv1D, find_pruneable_heads_and_indices, prune_conv1d_layer
from transformers.utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from transformers.models.decision_transformer.modeling_decision_transformer import DecisionTransformerOutput
from transformers.models.decision_transformer.configuration_decision_transformer import DecisionTransformerConfig
from transformers import DecisionTransformerPreTrainedModel, DecisionTransformerGPT2Model

class TDMPCDecisionTransformerModel(DecisionTransformerPreTrainedModel):
    """

    The model builds upon the GPT2 architecture to perform autoregressive prediction of actions in an offline RL
    setting. Refer to the paper for more details: https://arxiv.org/abs/2106.01345

    """

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.hidden_size = config.hidden_size
        # note: the only difference between this GPT2Model and the default Huggingface version
        # is that the positional embeddings are removed (since we'll add those ourselves)
        self.encoder = DecisionTransformerGPT2Model(config)

        self.embed_timestep = nn.Embedding(config.max_ep_len, config.hidden_size)
        self.embed_return = torch.nn.Linear(1, config.hidden_size)
        self.embed_state = torch.nn.Linear(config.state_dim, config.hidden_size)
        self.embed_action = torch.nn.Linear(config.act_dim, config.hidden_size)

        self.embed_ln = nn.LayerNorm(config.hidden_size)

        # note: we don't predict states or returns for the paper
        self.predict_state = torch.nn.Linear(config.hidden_size, config.state_dim)

        self.predict_q = torch.nn.Linear(config.hidden_size, 1)

        # woodwardbr: This is from tdmpc
        self.predict_action_and_log_prob = nn.Sequential(
            *([nn.Linear(config.hidden_size, 2*config.act_dim)])
        )

        self.predict_return = torch.nn.Linear(config.hidden_size, 1)

        # woodwardbr: Use this to have the data come in as (Horizon, Batchsize, Dimension)
        #             instead of (Batchsize, Horizon, Dimension)
        self.use_horizon_batchsize_dimensioning = config.use_horizon_batchsize_dimensioning


        # Initialize weights and apply final processing
        self.post_init()

    # add_start_docstrings_to_model_forward(DECISION_TRANSFORMER_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    # replace_return_docstrings(output_type=DecisionTransformerOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        states: Optional[torch.FloatTensor] = None,
        actions: Optional[torch.FloatTensor] = None,
        rewards: Optional[torch.FloatTensor] = None,
        returns_to_go: Optional[torch.FloatTensor] = None,
        timesteps: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        # task: Optional[torch.FloatTensor] = None, # TODO(woodwardbr): check if I can delete this
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = False,
    ) -> Union[Tuple[torch.FloatTensor], DecisionTransformerOutput]:
        r"""
        Returns:

        Examples:

        ```python
        >>> from transformers import DecisionTransformerModel
        >>> import torch

        >>> model = DecisionTransformerModel.from_pretrained("edbeeching/decision-transformer-gym-hopper-medium")
        >>> # evaluation
        >>> model = model.to(device)
        >>> model.eval()

        >>> env = gym.make("Hopper-v3")
        >>> state_dim = env.observation_space.shape[0]
        >>> act_dim = env.action_space.shape[0]

        >>> state = env.reset()
        >>> states = torch.from_numpy(state).reshape(1, 1, state_dim).to(device=device, dtype=torch.float32)
        >>> actions = torch.zeros((1, 1, act_dim), device=device, dtype=torch.float32)
        >>> rewards = torch.zeros(1, 1, device=device, dtype=torch.float32)
        >>> target_return = torch.tensor(TARGET_RETURN, dtype=torch.float32).reshape(1, 1)
        >>> timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)
        >>> attention_mask = torch.zeros(1, 1, device=device, dtype=torch.float32)

        >>> # forward pass
        >>> with torch.no_grad():
        ...     state_preds, action_preds, return_preds = model(
        ...         states=states,
        ...         actions=actions,
        ...         rewards=rewards,
        ...         returns_to_go=target_return,
        ...         timesteps=timesteps,
        ...         attention_mask=attention_mask,
        ...         return_dict=False,
        ...     )
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.use_horizon_batchsize_dimensioning:
            states = states.permute(1,0,2)
            actions = actions.permute(1,0,2)
            rewards = rewards.permute(1,0,2)
            returns_to_go = returns_to_go.permute(1,0,2)
            timesteps = timesteps.permute(1,0,2)
            if attention_mask is not None:
                attention_mask = attention_mask.permute(1,0)
            # tasks = tasks.permute(1,0,2)

        batch_size, seq_length = states.shape[0], states.shape[1]

        if attention_mask is None:
            # attention mask for GPT: 1 if can be attended to, 0 if not
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)

        # embed each modality with a different head
        state_embeddings = self.embed_state(states)
        action_embeddings = self.embed_action(actions)
        returns_embeddings = self.embed_return(returns_to_go)
        time_embeddings = self.embed_timestep(timesteps).reshape(batch_size,seq_length,self.hidden_size)

        # time embeddings are treated similar to positional embeddings
        state_embeddings = state_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings
        returns_embeddings = returns_embeddings + time_embeddings

        # this makes the sequence look like (R_1, s_1, a_1, R_2, s_2, a_2, ...)
        # which works nice in an autoregressive sense since states predict actions
        stacked_inputs = (
            torch.stack((returns_embeddings, state_embeddings, action_embeddings), dim=1)
            .permute(0, 2, 1, 3)
            .reshape(batch_size, 3 * seq_length, self.hidden_size)
        )
        stacked_inputs = self.embed_ln(stacked_inputs).to(self.device)

        # to make the attention mask fit the stacked inputs, have to stack it as well
        stacked_attention_mask = (
            torch.stack((attention_mask, attention_mask, attention_mask), dim=1)
            .permute(0, 2, 1)
            .reshape(batch_size, 3 * seq_length)
        ).to(self.device)
        device = self.device
        # we feed in the input embeddings (not word indices as in NLP) to the model
        encoder_outputs = self.encoder(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
            position_ids=torch.zeros(stacked_attention_mask.shape, device=device, dtype=torch.long),
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        x = encoder_outputs[0]

        # reshape x so that the second dimension corresponds to the original
        # returns (0), states (1), or actions (2); i.e. x[:,1,t] is the token for s_t
        x = x.reshape(batch_size, seq_length, 3, self.hidden_size).permute(0, 2, 1, 3)

        # get predictions
        return_preds = self.predict_return(x[:, 2])  # predict next return given state and action
        state_preds = self.predict_state(x[:, 2])  # predict next state given state and action
        q_preds = self.predict_q(x[:, 2]) # predict next Q given state and action

        # Gaussian policy prior
        action_preds, ac_log_std_preds = self.predict_action_and_log_prob(x[:, 1]).chunk(2, dim=-1)

        if self.use_horizon_batchsize_dimensioning:
            state_preds = state_preds.permute(1,0,2)
            q_preds = q_preds.permute(1,0,2)
            action_preds = action_preds.permute(1,0,2)
            ac_log_std_preds = ac_log_std_preds.permute(1,0,2)
            return_preds = return_preds.permute(1,0,2)

        # if not return_dict:
        return state_preds, q_preds, action_preds, ac_log_std_preds, return_preds

        # return DecisionTransformerOutput(
        #     last_hidden_state=encoder_outputs.last_hidden_state,
        #     state_preds=state_preds,
        #     action_preds=action_preds,
        #     return_preds=return_preds,
        #     hidden_states=encoder_outputs.hidden_states,
        #     attentions=encoder_outputs.attentions,
        # )


class TDMPCDecisionTransformerConfig(DecisionTransformerConfig):
    def __init__(
        self,
        num_tasks=1,
        use_horizon_batchsize_dimensioning=False,
        state_dim=17,
        act_dim=4,
        hidden_size=128,
        max_ep_len=4096,
        action_tanh=True,
        vocab_size=1,
        n_positions=1024,
        n_layer=3,
        n_head=1,
        n_inner=None,
        activation_function="relu",
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        scale_attn_weights=True,
        use_cache=True,
        bos_token_id=50256,
        eos_token_id=50256,
        scale_attn_by_inverse_layer_idx=False,
        reorder_and_upcast_attn=False,
        **kwargs,
    ):
        # For adding any additional KWArgs we want to config
        self.num_tasks = num_tasks
        self.use_horizon_batchsize_dimensioning = use_horizon_batchsize_dimensioning

        super().__init__(
            state_dim=state_dim, act_dim=act_dim,  max_ep_len=max_ep_len,
            action_tanh=action_tanh, vocab_size=vocab_size, hidden_size=hidden_size,
            n_positions=n_positions, _layer=n_layer, n_head=n_head,
            n_inner=n_inner, activation_function=activation_function,
            resid_pdrop=resid_pdrop, embd_pdrop=embd_pdrop, attn_pdrop=attn_pdrop,
            layer_norm_epsilon=layer_norm_epsilon, initializer_range=initializer_range,
            scale_attn_weights=scale_attn_weights, use_cache=use_cache,
            bos_token_id=bos_token_id, eos_token_id=eos_token_id,
            scale_attn_by_inverse_layer_idx=scale_attn_by_inverse_layer_idx,
            reorder_and_upcast_attn=reorder_and_upcast_attn, **kwargs)