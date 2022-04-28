import torch
import torch.nn as nn

import transformers


class MLPLayer(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, features):
        x = self.dense(features)
        x = self.activation(x)
        return x


class Similarity(nn.Module):
    def __init__(self, temp) -> None:
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp


class Pooler(nn.Module):
    def __init__(self, pooler_type) -> None:
        super().__init__()
        self.pool_type = pooler_type
        assert self.pool_type in [
            "cls",
            "cls_before_pooler",
            "avg",
            "avg_top2",
            "avg_first_last",
        ], ("unrecognized pooling type %s" % self.pool_type)

    def forward(self, attention_mask, outputs):
        last_hidden = outputs.last_hidden_state
        pooler_output = outputs.pooler_output
        hidden_states = outputs.hidden_states

        if self.pool_type in ["cls_before_pooler", "cls"]:
            return last_hidden[:, 0]

        elif self.pool_type == "avg":
            return (last_hidden * attention_mask.unsqueeze(-1)
                    ).sum(1) / attention_mask.sum(-1).unsqueeze(-1)

        elif self.pool_type == "avg_first_last":
            first_hidden = hidden_states[0]
            last_hidden = hidden_states[-1]
            pooled_result = ((first_hidden + last_hidden) / 2.0 *
                             attention_mask.unsqueeze(-1)
                             ).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result

        elif self.pool_type == "avg_top2":
            second_last_hidden = hidden_states[-2]
            last_hidden = hidden_states[-1]
            pooled_result = ((last_hidden + second_last_hidden) / 2.0 *
                             attention_mask.unsqueeze(-1)
                             ).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        else:
            raise NotImplementedError


def cl_init(cls, config):
    cls.pooler_type = cls.model_args.pooler_type
    cls.pooler = Pooler(cls.model_args.pooler_type)
    if cls.model_args.pooler_type == 'cls':
        cls.mlp = MLPLayer(config)

    cls.sim = Similarity(temp=cls.model_args.temp)
    cls.init_weights()
