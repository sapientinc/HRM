import torch

from .model_loader import load_model
from .utils import get_tokenizer, prepare_batch

def test_prediction(test_message: str):
    """
    指定されたメッセージでモデルの応答生成をテストする関数
    """
    print("--- CLIテスト開始 ---")
    
    # --- 1. デバイス、モデル、トークナイザの読み込み ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用デバイス: {device}")
    
    tokenizer = get_tokenizer()
    model, model_config = load_model(tokenizer=tokenizer, device=device)
    
    print(f"\nテストメッセージ: '{test_message}'")
    
    # --- 2. 応答生成の実行 ---
    try:
        # ユーザーのメッセージを受け取り、モデルの応答を生成する
        print("\n1. 入力テキストをトークンIDに変換中...")
        input_ids = tokenizer.encode(test_message)
        print(f" -> {input_ids}")

        print("\n2. モデル入力用のバッチを作成中...")
        batch = prepare_batch(input_ids, seq_len=model_config['seq_len'], device=device)

        print("\n3. モデルの初期状態 (carry) を作成中...")
        carry = model.initial_carry(batch)

        print("\n4. 推論実行中 (ACTループ開始)...")
        outputs = None
        step = 0
        while True:
            step += 1
            carry, _, _, current_outputs, all_finish = model(carry=carry, batch=batch, return_keys=['logits'])
            print(f" -> ステップ {step}: all_finish={all_finish}")
            if 'logits' in current_outputs:
                outputs = current_outputs
            if all_finish:
                print(" -> ホールティングシグナルを検知。ループ終了。")
                break
        
        print("\n5. 出力logitsからトークンIDを予測中...")
        if outputs and 'logits' in outputs:
            pred_logits = outputs['logits']
            pred_ids = torch.argmax(pred_logits, dim=-1).squeeze(0).cpu().tolist()
            
            print("\n6. 予測されたIDをテキストにデコード中...")
            response_text = tokenizer.decode(pred_ids)
            print(f" -> {pred_ids}")
        else:
            response_text = "モデルから有効な応答を得られませんでした。"

        print("\n--- 最終的なモデルの応答 ---")
        print(response_text)

    except Exception as e:
        print("\n--- エラーが発生しました ---")
        # スタックトレース全体を出力
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_prediction("hello world")
