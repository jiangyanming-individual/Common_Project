import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer

class TransformerClassifier(nn.Module):
    def __init__(self, num_classes):
        super(TransformerClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.fc(pooled_output)
        return logits

#加载模型
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# 示例输入
sentence = "This is an example sentence."
#进行编码
encoded_input = tokenizer.encode_plus(
    sentence,
    add_special_tokens=True,
    max_length=512,
    padding='max_length',
    truncation=True,
    return_tensors='pt'  # 返回PyTorch张量
)

input_ids = encoded_input['input_ids']
attention_mask = encoded_input['attention_mask']

num_classes = 2  # 分类的类别数量

model = TransformerClassifier(num_classes)
logits = model(input_ids, attention_mask)