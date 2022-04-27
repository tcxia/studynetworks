from requests import head
import torch
import torch.nn as nn
import transformers
from transformers import (
    RobertaTokenizer,
    RobertaForTokenClassification,
    RobertaConfig,
    RobertaModel,
    RobertaForMaskedLM,
)


class RobertaNER(RobertaForTokenClassification):
    config_class = RobertaConfig
    base_model_prefix = "roberta"

    def __init__(
        self,
        config,
        support_per_class,
        cuda_device,
        use_bias=True,
        use_global=False,
        dataset_label_nums=None,
    ):
        super().__init__(config)

        self.support_per_class = support_per_class
        self.cuda_device = cuda_device
        self.roberta = RobertaModel(config)
        self.use_global = use_global
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.use_bias = use_bias
        self.background = nn.Parameter(torch.zeros(1) + 70.0, requires_grad=True)
        self.class_metric = nn.Parameter(
            torch.ones(dataset_label_nums) + 0.0, requires_grad=True
        )
        self.classifiers = []
        if self.use_global:
            for i in range(len(dataset_label_nums)):
                self.classifiers.append(
                    nn.Linear(config.hidden_size, dataset_label_nums)
                )

        self.layer1 = nn.Sequential(nn.Linear(config.hidden_size, 128), nn.ReLU())
        self.layer2 = nn.Linear(128, 1)
        nn.init.xavier_uniform_(self.layer2.weight)

        self.alpha = nn.Parameter(torch.Tensor(1))
        nn.init.constant_(self.alpha, 0)

        self.beta = nn.Parameter(torch.Tensor(1))
        nn.init.constant_(self.beta, 0)

        self.init_weights()

    def compute_prototypes(
        self,
        input_ids,
        support_class_num,
        ori_prototypes=None,
        ori_embed_class_len=None,
        attention_mask=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
    ):
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        support_sets = outputs[0]
        batch_size, max_len, feat_dim = support_sets.shape

        if self.use_bias:
            embeds_per_class = [[] for _ in range(support_class_num * 2)]
            embeds_class_len = [0 for _ in range(support_class_num * 2)]

        else:
            embeds_per_class = [[] for _ in range(support_class_num * 2 + 1)]
            embeds_class_len = [0 for _ in range(support_class_num * 2 + 1)]

        labels_numpy = labels.data

        for i_sen, sentence in enumerate(support_sets):
            for i_word, word in enumerate(sentence):
                if attention_mask[i_sen, i_word] == 1:
                    tag = labels_numpy[i_sen][i_word]
                    if tag > 0 and self.use_bias:
                        embeds_per_class[tag - 1].append(word)
                    if tag >= 0 and self.use_bias == False:
                        embeds_per_class[tag].append(word)

        prototypes = [torch.zeros_like(embeds_per_class[0][0]) for _ in range(len(embeds_per_class))]
        
        for i in range(len(embeds_per_class)):
            if ori_embed_class_len is not None:
                embeds_class_len[i] = len(embeds_per_class[i]) + ori_embed_class_len[i]
            else:
                embeds_class_len[i] = len(embeds_per_class[i])
            
            if ori_prototypes is not None and embeds_class_len[i] > 0:
                prototypes[i] += ori_prototypes[i] * ori_embed_class_len[i] / embeds_class_len[i]

            for embed in embeds_per_class[i]:
                prototypes[i] += embed / embeds_class_len[i]


        prototypes = torch.cat([x.unsqueeze(0) for x in prototypes])

        return prototypes, embeds_class_len

    def instance_scale(self, input):
        sigma = self.layer1(input)
        sigma = self.layer2(sigma)
        sigma = torch.sigmoid(sigma)
        sigma = torch.exp(self.alpha) * sigma + torch.exp(self.beta)
        return sigma