"""
from transformers import BertModel, BertTokenizer, AutoModel, AutoTokenizer

### for vi
bert = AutoModel.from_pretrained("VoVanPhuc/sup-SimCSE-VietNamese-phobert-base")
tokenizer = AutoTokenizer.from_pretrained("VoVanPhuc/sup-SimCSE-VietNamese-phobert-base")
bert.save_pretrained("models/sim_cse_phobert/bert")
tokenizer.save_pretrained("models/sim_cse_phobert/tokenizer")

### for en
bert = AutoModel.from_pretrained("bert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
bert.save_pretrained("models/bert_base_uncased/bert")
tokenizer.save_pretrained("models/bert_base_uncased/tokenizer")
"""