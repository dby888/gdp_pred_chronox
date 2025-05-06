import pandas as pd
import torch
from torch.utils.data import Dataset, random_split
from transformers import T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments, EarlyStoppingCallback
import numpy as np

# 定义 bins：-15 到 15，每 0.1 一格
bins = np.arange(-15, 15.1, 0.1)
labels = list(range(len(bins) - 1))

def bin_transform(values):
    values = np.array(values)
    bin_ids = np.digitize(values, bins) - 1
    bin_ids = np.clip(bin_ids, 0, len(labels) - 1)
    return bin_ids

def inverse_transform(bin_ids):
    bin_ids = np.array(bin_ids)
    pred_values = (bins[bin_ids] + bins[bin_ids + 1]) / 2
    return pred_values

# 加载模型和 tokenizer
model_path = 'D:/huggingFaceModels/chronos-t5-large'
model = T5ForConditionalGeneration.from_pretrained(model_path)
tokenizer = T5Tokenizer.from_pretrained('t5-small')

# 改成分类 head
model.lm_head = torch.nn.Linear(model.config.d_model, len(labels))
model.config.num_labels = len(labels)

# 冻结主模型参数，只训练新 head
for name, param in model.named_parameters():
    if not name.startswith('lm_head') and not name.startswith('encoder.iib_'):
        param.requires_grad = False

# 加载 NFCI 数据
nfci_df = pd.read_csv('NFCI.csv', parse_dates=['observation_date'])
nfci_df['Year'] = nfci_df['observation_date'].dt.year
nfci_df['Month'] = nfci_df['observation_date'].dt.month
nfci_q1 = nfci_df[(nfci_df['Month'] <= 3) & (nfci_df['Year'] >= 1980) & (nfci_df['Year'] <= 2024)]
nfci_yearly = nfci_q1.groupby('Year')['NFCI'].mean()

# Dataset 类
class GDPDataset(Dataset):
    def __init__(self, excel_file, country, target_subject, covariate_series, tokenizer, max_length=128):
        df = pd.read_excel(excel_file, skiprows=1)
        df.columns = df.columns.astype(str)
        country_data = df[(df['ISO'] == country) & (df['WEO Subject Code'] == target_subject)]
        years = [str(y) for y in range(1980, 2025)]

        self.texts = []
        self.targets = []

        for year in years:
            if year in country_data.columns:
                value = country_data[year].values[0]
                cov_value = covariate_series.get(int(year), 'NA')

                if pd.notna(value):
                    prompt = f"Predict {target_subject} for {country} in {year} with NFCI={cov_value}"
                    self.texts.append(prompt)
                    self.targets.append(float(value))

        self.encodings = tokenizer(self.texts, truncation=True, padding=True, max_length=max_length)
        self.targets_enc = bin_transform(self.targets)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.targets_enc[idx], dtype=torch.long)
        return item

# 加载数据集
dataset = GDPDataset(
    excel_file='copieofWEO2024.xlsx',
    country='USA',
    target_subject='NGDP_RPCH',
    covariate_series=nfci_yearly,
    tokenizer=tokenizer
)

# 拆分 80/20
train_size = int(0.8 * len(dataset))
valid_size = len(dataset) - train_size
train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])

# 训练参数
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=10,
    per_device_train_batch_size=4,
    learning_rate=1e-5,
    weight_decay=0.01,
    save_total_limit=2,
    logging_dir='./logs',
    logging_steps=10,
    metric_for_best_model="loss",
    eval_strategy="steps",
    save_strategy="steps",
    eval_steps=100,
    save_steps=100,
    load_best_model_at_end=True,
)
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss

# 自定义 compute_loss
def compute_loss(model, inputs, return_outputs=False):
    labels = inputs.pop("labels")
    outputs = model(**inputs)
    logits = outputs.logits
    loss_fct = torch.nn.CrossEntropyLoss()
    loss = loss_fct(logits, labels)
    return (loss, outputs) if return_outputs else loss

# Trainer
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

# 🚨 检查 batch
train_dataloader = trainer.get_train_dataloader()
for batch in train_dataloader:
    for key, val in batch.items():
        if torch.is_tensor(val):
            if torch.isnan(val).any():
                print(f"❌ NaN detected in {key}")
            if torch.isinf(val).any():
                print(f"❌ Inf detected in {key}")
    if "labels" in batch:
        labels = batch["labels"]
        print(f"Labels min: {labels.min()}, max: {labels.max()}")
    break

# 开始训练
trainer.train()

# 保存微调模型
save_path = './finetuned-chronos-t5'
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
print(f"✅ Finetuned model saved to {save_path}")
