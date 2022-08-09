import torch
import pandas as pd
from model import BERT, nn
from transformers import AdamW
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from torch.optim.lr_scheduler import StepLR
import warnings
from colorama import Fore
from colorama import Style
from transformers import AutoModel, AutoTokenizer, BertModel, BertTokenizer
import py_vncorenlp
from paths import VNCORE_ABS_PATH, VI_MODEL, VI_TOKENIZER, EN_MODEL, EN_TOKENIZER

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# load dataset
lang = ""
while True:
    lang = input("lang=").strip().lower()
    if lang not in ["en", "vi"]:
        print(f"{Fore.RED}lang must be either 'en' or 'vi'!{Style.RESET_ALL}!")
        continue
    break

# define batch size, epochs
batch_size = 16
epochs = 200
while True:
    try:
        batch_size, epochs = [
            int(val.strip()) for val in input("batch size, epoch (default=16 200)=").strip().split(r" ")
        ]
        break
    except ValueError:
        print(
            f"{Fore.RED}batch size and epoch are int! input example: 16 200{Style.RESET_ALL}"
        )
        continue

# set learning rate
lr = 0.0001
while True:
    try:
        lr = float(input("lr (default=0.0001)="))
        break
    except ValueError:
        print(f"{Fore.RED}learning rate should be float and small!{Style.RESET_ALL}!")
        continue


data = pd.read_csv(f"dataset/{lang}.csv")

# prepare data
le = LabelEncoder()
data["label"] = le.fit_transform(data["label"])
data["label"].value_counts(normalize=True)
train_text, train_labels = data["text"], data["label"]

# load model for specific lang
if lang == "vi":
    bert = AutoModel.from_pretrained(VI_MODEL)
    tokenizer = AutoTokenizer.from_pretrained(VI_TOKENIZER)
    rdr_segmenter = py_vncorenlp.VnCoreNLP(
        annotators=["wseg"],
        save_dir=VNCORE_ABS_PATH
    )
    train_text_list = train_text.tolist()
    train_text_list = [rdr_segmenter.word_segment(text)[0] for text in train_text_list]
    tokens_train = tokenizer(
        train_text_list,
        padding=True,
        truncation=True,
        return_token_type_ids=False,
    )
else:
    bert = BertModel.from_pretrained(EN_MODEL)
    tokenizer = BertTokenizer.from_pretrained(EN_TOKENIZER)
    tokens_train = tokenizer(
        train_text.tolist(),
        padding=True,
        truncation=True,
        return_token_type_ids=False,
    )

train_seq = torch.tensor(tokens_train["input_ids"])
train_mask = torch.tensor(tokens_train["attention_mask"])
train_y = torch.tensor(train_labels.tolist())

# wrap tensors
train_data = TensorDataset(train_seq, train_mask, train_y)
# sampler for sampling the data during training
train_sampler = RandomSampler(train_data)
# DataLoader for train set
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)


# use gpu if available
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


# model
for param in bert.parameters():
    param.requires_grad = False
model = BERT(bert)
model = model.to(device)

optimizer = AdamW(params=model.parameters(), lr=lr)


class_wts = compute_class_weight(
    class_weight="balanced", classes=np.unique(train_labels), y=train_labels
)


weights = torch.tensor(class_wts, dtype=torch.float)
weights = weights.to(device)
cross_entropy = nn.NLLLoss(weight=weights)
train_losses = []


lr_sch = StepLR(optimizer, step_size=1, gamma=0.1)


def train():
    model.train()
    total_loss = 0
    total_preds = []
    for step, batch in enumerate(train_dataloader):
        if step % 50 == 0 and not step == 0:
            print(" Batch {:>5,} of {:>5,}.".format(step, len(train_dataloader)))
        batch = [r.to(device) for r in batch]
        sent_id, mask, labels = batch
        preds = model(sent_id, mask)
        loss = cross_entropy(preds, labels)
        total_loss = total_loss + loss.item()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
        preds = preds.detach().cpu().numpy()
        total_preds.append(preds)
    avg_loss = total_loss / len(train_dataloader)
    total_preds = np.concatenate(total_preds, axis=0)
    return avg_loss, total_preds


for epoch in range(epochs):
    print("Epoch {:} / {:}".format(epoch + 1, epochs))
    train_loss, _ = train()
    train_losses.append(train_loss)
    print(f"Loss: {train_loss:.3f}\n")


torch.save(model.state_dict(), f"./output_weights/{lang}_model_weight.pth")
