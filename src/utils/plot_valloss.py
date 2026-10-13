import matplotlib.pyplot as plt
import re
import sys
import os
from pathlib import Path
import numpy as np
import argparse

def parse_valloss_file(file_path):
    """Parse the valloss.txt file to extract steps, RC and KL values from validation data."""
    steps = []
    rc_values = []
    kl_values = []
    
    with open(file_path, 'r') as file:
        for line in file:
            # Extract [val] RC and KL values: [step XXXX] ... [val] | RC: X.XXXX | KL: X.XXXX
            match = re.search(r'\[step (\d+)\].*\[val\] \| RC: ([\d.]+) \| KL: ([\d.\-]+)', line)
            if match:
                steps.append(int(match.group(1)))
                rc_values.append(float(match.group(2)))
                kl_values.append(float(match.group(3)))
    
    return steps, rc_values, kl_values

# コマンドライン引数のパース
parser = argparse.ArgumentParser(description='検証ログから損失曲線をグラフ化します')
parser.add_argument('--log', '-l', type=str, nargs='+', 
                    help='ログファイルのパス（複数指定可）')
parser.add_argument('--output', '-o', type=str, default='.',
                    help='グラフ保存先ディレクトリ（デフォルト: カレントディレクトリ）')
args = parser.parse_args()

# 出力ディレクトリが存在しなければ作成
os.makedirs(args.output, exist_ok=True)

# ログファイルのバリデーション
if not args.log:
    print("エラー: ログファイルを指定してください (-l オプションを使用)")
    sys.exit(1)

log_files = args.log if isinstance(args.log, list) else [args.log]

# ログファイルの存在確認と読み込み
dataframes = {}
for log_file in log_files:
    if not os.path.exists(log_file):
        print(f"エラー: ログファイルが見つかりません: {log_file}")
        sys.exit(1)
    steps, rc_values, kl_values = parse_valloss_file(log_file)
    if not steps:
        print(f"警告: {log_file} から有効なデータが取得できませんでした")
        continue
    dataframes[log_file] = {
        'steps': steps,
        'rc_values': rc_values,
        'kl_values': kl_values
    }

if not dataframes:
    print("エラー: 有効なログファイルが見つかりません")
    sys.exit(1)

# ========================================
# 複数ファイルの場合：比較グラフを生成
# ========================================
if len(dataframes) > 1:
    print(f"\n複数ファイルモード: {len(dataframes)}個のログファイルを比較します")
    
    # カラーマップを設定
    colors = plt.cm.tab10(np.linspace(0, 1, len(dataframes)))
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Validation Loss Comparison Across Multiple Runs', fontsize=16)
    
    # 再構成損失（RC）の比較
    for idx, (log_file, data) in enumerate(dataframes.items()):
        # ファイルのレーベルを生成（ディレクトリ名またはファイル名から）
        if 'checkpoints' in log_file:
            label = os.path.basename(os.path.dirname(log_file))
        else:
            label = os.path.basename(log_file)
        axes[0].plot(data['steps'], data['rc_values'], linewidth=2, 
                    label=label, color=colors[idx], marker='o', markersize=4, markevery=10)
    
    axes[0].set_xlabel('Steps', fontsize=12)
    axes[0].set_ylabel('Validation RC Loss', fontsize=12)
    axes[0].set_title('Validation Reconstruction Loss Comparison', fontsize=14)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=10)
    
    # KL発散損失の比較
    for idx, (log_file, data) in enumerate(dataframes.items()):
        # ファイルのレーベルを生成（ディレクトリ名またはファイル名から）
        if 'checkpoints' in log_file:
            label = os.path.basename(os.path.dirname(log_file))
        else:
            label = os.path.basename(log_file)
        axes[1].plot(data['steps'], data['kl_values'], linewidth=2, 
                    label=label, color=colors[idx], marker='s', markersize=4, markevery=10)
    
    axes[1].set_xlabel('Steps', fontsize=12)
    axes[1].set_ylabel('Validation KL Loss', fontsize=12)
    axes[1].set_title('Validation KL Divergence Loss Comparison', fontsize=14)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=10)
    
    plt.tight_layout()
    output_path_compare = os.path.join(args.output, 'valloss_comparison.png')
    plt.savefig(output_path_compare, dpi=300, bbox_inches='tight')
    print(f"比較グラフを '{output_path_compare}' に保存しました")
    plt.close()
else:
    # 単一ファイルの場合：グラフを生成
    print(f"\n単一ファイルモード: 検証ログから学習曲線を生成します")
    
    log_file = list(dataframes.keys())[0]
    data = dataframes[log_file]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Validation Loss Curves', fontsize=16)
    
    # 再構成損失（RC）のグラフ
    axes[0].plot(data['steps'], data['rc_values'], linewidth=2, 
                label='RC Loss', color='#1f77b4', marker='o', markersize=4, markevery=10)
    axes[0].set_xlabel('Steps', fontsize=12)
    axes[0].set_ylabel('Validation RC Loss', fontsize=12)
    axes[0].set_title('Validation Reconstruction Loss', fontsize=14)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=11)
    
    # KL発散損失のグラフ
    axes[1].plot(data['steps'], data['kl_values'], linewidth=2, 
                label='KL Loss', color='#ff7f0e', marker='s', markersize=4, markevery=10)
    axes[1].set_xlabel('Steps', fontsize=12)
    axes[1].set_ylabel('Validation KL Loss', fontsize=12)
    axes[1].set_title('Validation KL Divergence Loss', fontsize=14)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=11)
    
    plt.tight_layout()
    output_path_single = os.path.join(args.output, 'valloss_curve.png')
    plt.savefig(output_path_single, dpi=300, bbox_inches='tight')
    print(f"学習曲線を '{output_path_single}' に保存しました")
    plt.close()