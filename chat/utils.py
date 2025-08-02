import json
import os
import torch

# シンプルな文字レベルのトークナイザ（プレースホルダー）
class SimpleCharTokenizer:
    def __init__(self, vocab=None):
        if vocab:
            self.char_to_id = vocab
            self.id_to_char = {i: c for c, i in vocab.items()}
        else:
            # デフォルトの基本的な語彙
            chars = sorted(list(set('abcdefghijklmnopqrstuvwxyz0123456789 ,.?!\n')))
            self.char_to_id = {c: i for i, c in enumerate(chars)}
            self.id_to_char = {i: c for i, c in enumerate(chars)}
        
        self.vocab_size = len(self.char_to_id)

    def encode(self, text):
        return [self.char_to_id.get(c, 0) for c in text.lower()]

    def decode(self, ids):
        return "".join([self.id_to_char.get(i, '') for i in ids])

def get_tokenizer():
    """
    vocab.json が存在すればそれを読み込み、なければプレースホルダートークナイザを返す
    """
    vocab_path = os.path.join(os.path.dirname(__file__), 'vocab.json')
    if os.path.exists(vocab_path):
        with open(vocab_path, 'r') as f:
            vocab = json.load(f)
        print("Loaded tokenizer from vocab.json")
        return SimpleCharTokenizer(vocab)
    else:
        print("Warning: vocab.json not found. Using a placeholder tokenizer.")
        return SimpleCharTokenizer()

def prepare_batch(input_ids, seq_len, device="cpu"):
    """
    モデルに入力するためのバッチを作成する
    """
    # パディングまたはトランケーション
    if len(input_ids) > seq_len:
        input_ids = input_ids[:seq_len]
    else:
        input_ids = input_ids + [0] * (seq_len - len(input_ids))

    # バッチディクショナリの作成
    batch = {
        "inputs": torch.tensor([input_ids], dtype=torch.long, device=device),
        "puzzle_identifiers": torch.tensor([0], dtype=torch.long, device=device), # ダミー
        "targets": torch.tensor([input_ids], dtype=torch.long, device=device), # 推論時は不要だが、モデルによっては必要
        "labels": torch.tensor([input_ids], dtype=torch.long, device=device) # labelsキーを追加
    }
    return batch
