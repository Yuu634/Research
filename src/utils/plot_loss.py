import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

# コマンドライン引数のパース
parser = argparse.ArgumentParser(description='トレーニングログから損失曲線をグラフ化します')
parser.add_argument('--log', '-l', type=str, nargs='+', 
                    default=['checkpoints_drum_cp_with_difficulty/log.txt'],
                    help='ログファイルのパス（複数指定可。デフォルト: checkpoints_drum_cp_with_difficulty/log.txt）')
parser.add_argument('--output', '-o', type=str, default='.',
                    help='グラフ保存先ディレクトリ（デフォルト: カレントディレクトリ）')
args = parser.parse_args()

# 出力ディレクトリが存在しなければ作成
os.makedirs(args.output, exist_ok=True)

# ログファイルを読み込む
log_files = args.log if isinstance(args.log, list) else [args.log]
dataframes = {}
for log_file in log_files:
    if not os.path.exists(log_file):
        print(f"エラー: ログファイルが見つかりません: {log_file}")
        exit(1)
    dataframes[log_file] = pd.read_csv(log_file, sep=r'\s+')

# 単一ファイルの場合のメイン処理用
df = dataframes[log_files[0]]

# 日本語フォント設定（オプション）
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']

# ========================================
# 複数ファイルの場合：比較グラフを生成
# ========================================
if len(log_files) > 1:
    print(f"\n複数ファイルモード: {len(log_files)}個のログファイルを比較します")
    
    # カラーマップを設定
    colors = plt.cm.tab10(np.linspace(0, 1, len(log_files)))
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Training Loss Comparison Across Multiple Runs', fontsize=16)
    
    # 再構成損失の比較
    for idx, (log_file, df_log) in enumerate(dataframes.items()):
        # ファイルのレーベルを生成（ディレクトリ名またはファイル名から）
        if 'checkpoints' in log_file:
            label = os.path.basename(os.path.dirname(log_file))
        else:
            label = os.path.basename(log_file)
        axes[0].plot(df_log['steps'], df_log['recons_loss'], linewidth=2, 
                    label=label, color=colors[idx], marker='o', markersize=3, markevery=10)
    
    axes[0].set_xlabel('Steps', fontsize=12)
    axes[0].set_ylabel('Reconstruction Loss', fontsize=12)
    axes[0].set_title('Reconstruction Loss Comparison', fontsize=14)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=10)
    
    # KL発散損失の比較
    for idx, (log_file, df_log) in enumerate(dataframes.items()):
        # ファイルのレーベルを生成（ディレクトリ名またはファイル名から）
        if 'checkpoints' in log_file:
            label = os.path.basename(os.path.dirname(log_file))
        else:
            label = os.path.basename(log_file)
        axes[1].plot(df_log['steps'], df_log['kldiv_loss'], linewidth=2, 
                    label=label, color=colors[idx], marker='s', markersize=3, markevery=10)
    
    axes[1].set_xlabel('Steps', fontsize=12)
    axes[1].set_ylabel('KL Divergence Loss', fontsize=12)
    axes[1].set_title('KL Divergence Loss Comparison', fontsize=14)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=10)
    
    plt.tight_layout()
    output_path_compare = os.path.join(args.output, 'loss_comparison.png')
    plt.savefig(output_path_compare, dpi=300, bbox_inches='tight')
    print(f"比較グラフを '{output_path_compare}' に保存しました")
    plt.show()
    exit(0)

# ========================================
# 単一ファイルの場合：従来の詳細グラフを生成
# ========================================
print(f"\n単一ファイルモード: {log_files[0]}")

# 図を作成
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Training Loss Curves', fontsize=16)

# 1. 再構成損失（Reconstruction Loss）
axes[0, 0].plot(df['steps'], df['recons_loss'], linewidth=2, color='blue', label='recons_loss')
axes[0, 0].set_xlabel('Steps')
axes[0, 0].set_ylabel('Reconstruction Loss')
axes[0, 0].set_title('Reconstruction Loss')
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].legend()

# 2. KL Divergence Loss
axes[0, 1].plot(df['steps'], df['kldiv_loss'], linewidth=2, color='red', label='kldiv_loss')
axes[0, 1].set_xlabel('Steps')
axes[0, 1].set_ylabel('KL Divergence Loss')
axes[0, 1].set_title('KL Divergence Loss')
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].legend()

# 3. KL Divergence Raw
axes[1, 0].plot(df['steps'], df['kldiv_raw'], linewidth=2, color='green', label='kldiv_raw')
axes[1, 0].set_xlabel('Steps')
axes[1, 0].set_ylabel('KL Divergence Raw')
axes[1, 0].set_title('KL Divergence Raw')
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].legend()

# 4. すべての損失を重ねて表示（正規化版）
recons_norm = (df['recons_loss'] - df['recons_loss'].min()) / (df['recons_loss'].max() - df['recons_loss'].min())
kldiv_norm = (df['kldiv_loss'] - df['kldiv_loss'].min()) / (df['kldiv_loss'].max() - df['kldiv_loss'].min())

axes[1, 1].plot(df['steps'], recons_norm, linewidth=2, label='recons_loss (normalized)', color='blue')
axes[1, 1].plot(df['steps'], kldiv_norm, linewidth=2, label='kldiv_loss (normalized)', color='red')
axes[1, 1].set_xlabel('Steps')
axes[1, 1].set_ylabel('Normalized Loss')
axes[1, 1].set_title('All Losses (Normalized)')
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].legend()

plt.tight_layout()
output_path1 = os.path.join(args.output, 'loss_curves.png')
plt.savefig(output_path1, dpi=300, bbox_inches='tight')
print(f"グラフを '{output_path1}' に保存しました")
plt.show()

# エポック別のプロット
fig2, ax = plt.subplots(figsize=(12, 6))
epochs = df['ep'].unique()
for epoch in epochs:
    epoch_data = df[df['ep'] == epoch]
    ax.plot(epoch_data['steps'], epoch_data['recons_loss'], marker='o', label=f'Epoch {epoch}', markersize=3)

ax.set_xlabel('Steps')
ax.set_ylabel('Reconstruction Loss')
ax.set_title('Reconstruction Loss by Epoch')
ax.grid(True, alpha=0.3)
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
output_path2 = os.path.join(args.output, 'loss_by_epoch.png')
plt.savefig(output_path2, dpi=300, bbox_inches='tight')
print(f"エポック別グラフを '{output_path2}' に保存しました")
plt.show()
