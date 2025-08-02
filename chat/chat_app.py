import gradio as gr
import torch
import os


from chat.model_loader import load_model
from chat.utils import get_tokenizer, prepare_batch

# --- グローバル変数の設定 ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# --- モデルとトークナイザの読み込み ---
# アプリケーション起動時に一度だけ実行
TOKENIZER = get_tokenizer()
MODEL, MODEL_CONFIG = load_model(tokenizer=TOKENIZER, device=DEVICE)


# --- チャットボットの応答生成関数 ---
@torch.inference_mode()
def predict(message, history):
    """
    ユーザーのメッセージを受け取り、モデルの応答を生成する関数
    """
    # 1. 入力テキストをトークンIDに変換
    input_ids = TOKENIZER.encode(message)

    # 2. モデル入力用のバッチを作成
    batch = prepare_batch(input_ids, seq_len=MODEL_CONFIG['seq_len'], device=DEVICE)

    # 3. モデルの初期状態 (carry) を作成
    carry = MODEL.initial_carry(batch)

    # 4. 推論実行 (ACTにより内部でループ)
    outputs = None
    while True:
        carry, _, _, current_outputs, all_finish = MODEL(carry=carry, batch=batch, return_keys=['logits'])
        if 'logits' in current_outputs:
            outputs = current_outputs
        if all_finish:
            break
    
    # 5. 出力logitsからトークンIDを予測
    if outputs and 'logits' in outputs:
        pred_logits = outputs['logits']
        pred_ids = torch.argmax(pred_logits, dim=-1).squeeze(0).cpu().tolist()
        # 6. 予測されたIDをテキストにデコード
        response_text = TOKENIZER.decode(pred_ids)
    else:
        response_text = "モデルから有効な応答を得られませんでした。"

    # 簡単な後処理
    response_text = response_text.strip()

    return response_text

# --- Gradio UI の作成と起動 ---
if __name__ == "__main__":
    # Gradioチャットインターフェースを作成
    chat_interface = gr.ChatInterface(
        fn=predict,
        title="HRM Chat",
        description="Hierarchical Reasoning Model とチャットします。メッセージを入力してEnterを押してください。",
        chatbot=gr.Chatbot(height=600),
        textbox=gr.Textbox(placeholder="ここにメッセージを入力...", container=False, scale=7),
        type="messages"
    )

    # アプリケーションを起動
    print("Starting Gradio app...")
    chat_interface.launch(server_name="127.0.0.1")
