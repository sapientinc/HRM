import os
import torch
from omegaconf import OmegaConf
import yaml

from utils.functions import load_model_class
from models.losses import ACTLossHead

def load_model(tokenizer, device="cpu"):
    """
    設定ファイルからモデルアーキテクチャを読み込み、初期化する。
    学習済みチェックポイントが存在すれば、それを読み込む。
    """
    # --- 1. 設定ファイルの読み込み ---
    config_path = os.path.join('config', 'arch', 'hrm_v1.yaml')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, 'r') as f:
        arch_config_dict = yaml.safe_load(f)
    
    arch_config = OmegaConf.create(arch_config_dict)

    # --- 2. モデル設定の構築 ---
    # OmegaConfの補間を解決してから辞書に変換
    resolved_config = OmegaConf.to_container(arch_config, resolve=True)

    model_cfg = dict(
        **resolved_config,
        batch_size=1,
        vocab_size=tokenizer.vocab_size,
        seq_len=256,
        num_puzzle_identifiers=1,
        causal=False
    )
    # 不要なキーを削除
    model_cfg.pop('name', None)
    model_cfg.pop('loss', None)

    # --- 3. モデルのインスタンス化 ---
    print(f"Loading model: {arch_config.name}")
    model_cls = load_model_class(arch_config.name)
    loss_head_cls = load_model_class(arch_config.loss.name)

    # 損失ヘッドの初期化時に不要な 'name' キーを削除
    loss_config = OmegaConf.to_container(arch_config.loss, resolve=True)
    loss_config.pop('name', None)

    inner_model = model_cls(model_cfg)
    model = loss_head_cls(inner_model, **loss_config)

    # --- 4. チェックポイントの読み込み ---
    checkpoint_path = os.path.join(os.path.dirname(__file__), 'model_checkpoint.pth')
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from: {checkpoint_path}")
        # ACTLossHeadでラップされているため、model.modelにstate_dictをロードする
        # 保存方法によってキーが異なる場合があるため、柔軟に対応
        state_dict = torch.load(checkpoint_path, map_location=device)
        try:
            model.load_state_dict(state_dict)
        except RuntimeError:
            print("Could not load state_dict directly. Trying to load into model.model...")
            model.model.load_state_dict(state_dict)

    else:
        print("Warning: Model checkpoint not found. Using randomly initialized weights.")

    model.to(device)
    model.eval()

    print("Model loaded successfully.")
    return model, model_cfg
