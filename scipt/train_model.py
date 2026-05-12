import datetime
import argparse
import os
import sys
import subprocess
import re

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dataset import *
from model import *
from utils import *
# 自动创建 logs 文件夹（不存在就新建）
default_log_dir = "/CMAPSS-release/logs"
if not os.path.exists(default_log_dir):
    os.makedirs(default_log_dir)


DEFAULT_SEEDS = [2, 17, 27, 30, 33, 51, 62, 80, 88, 97]


def _build_log_name(model_code, lr, embed_dim, topk, time_tag):
    model_code = str(model_code).strip()
    if not model_code:
        raise ValueError("--model-code cannot be empty.")
    lr_str = format(float(lr), "g")
    return (
        f"{model_code}_学习率+{lr_str}_嵌入模型维度embed+{embed_dim}_"
        f"topk+{topk}_{time_tag}.log"
    )


ABLATION_CODE_CONFIGS = {
    'A': {
        'description': 'KNN graph + GAT-LSTM with encoder and decoder (AEF on)',
        'overrides': {
            'model_structure': 'encoderdecoder',
            'graph_mode': 'dynamic_knn',
            'use_spatial_gat': True,
            'use_decoder': True,
            'use_aef': True,
        },
    },
    'B': {
        'description': 'Full-connected graph + GAT-LSTM with encoder and decoder (AEF on)',
        'overrides': {
            'model_structure': 'encoderdecoder',
            'graph_mode': 'dynamic_knn',
            'use_spatial_gat': True,
            'use_decoder': True,
            'use_aef': True,
        },
    },
    'C': {
        'description': 'KNN graph + GAT-LSTM with encoder only (no decoder)',
        'overrides': {
            'model_structure': 'encoderdecoder',
            'graph_mode': 'dynamic_knn',
            'use_spatial_gat': True,
            'use_decoder': False,
            'use_aef': False,
        },
    },
    'D': {
        'description': 'Original GAT-LSTM without encoder and decoder',
        'overrides': {
            'model_structure': 'original',
            'graph_mode': 'path',
            'use_aef': False,
        },
    },
}


def _extract_ablation_code(model_code):
    code = str(model_code).strip().upper()
    if code in ABLATION_CODE_CONFIGS:
        return code

    tokens = [token for token in re.split(r'[^A-Z0-9]+', code) if token]
    for token in reversed(tokens):
        if token in ABLATION_CODE_CONFIGS:
            return token

    return None


def _apply_ablation_preset_from_code(args, model_code):
    ablation_code = _extract_ablation_code(model_code)
    if ablation_code is None:
        return None

    preset = ABLATION_CODE_CONFIGS[ablation_code]
    for key, value in preset['overrides'].items():
        setattr(args, key, value)

    if ablation_code == 'B':
        # Full-connected sensor graph via dynamic_knn with k=num_sensors.
        args.gat_topk = int(args.feature_num)

    return ablation_code, preset['description']


def _build_model_by_structure(args):
    model_structure = str(args.model_structure).strip().lower()

    if model_structure == 'encoderdecoder':
        if not args.use_decoder:
            args.use_aef = False

        encoder_input_size = args.gat_out_dim if args.use_spatial_gat else args.feature_num
        encoder = Seq2SeqEncoder(input_size=encoder_input_size, num_layers=2, num_hidden=8)

        decoder = None
        if args.use_decoder:
            decoder = Seq2SeqDecoder(
                input_size=encoder_input_size,
                num_layers=2,
                num_hidden=8,
                seq_len=args.sequence_len,
                attention_size=args.decoder_attention_size,
                use_aef=args.use_aef,
            )

        return EncoderDecoder(
            encoder=encoder,
            decoder=decoder,
            use_spatial_gat=args.use_spatial_gat,
            graph_mode=args.graph_mode,
            num_sensors=args.feature_num,
            gat_hidden_dim=args.gat_hidden_dim,
            gat_out_dim=args.gat_out_dim,
            gat_num_layers=args.gat_num_layers,
            gat_embed_dim=args.gat_embed_dim,
            gat_topk=args.gat_topk,
            gat_dropout=args.gat_dropout,
            gat_alpha=args.gat_alpha,
            use_decoder=args.use_decoder,
        )

    if not args.use_spatial_gat:
        raise ValueError(
            "--disable-spatial-gat is only supported by encoderdecoder structure. "
            "lstm/original structures always use graph-enhanced inputs."
        )

    gat_hidden_dims = [args.gat_hidden_dim, args.gat_out_dim]
    lstm_hidden_dims = [args.lstm_hidden_dim]

    if model_structure == 'original':
        return GAT_LSTM_model(
            num_patch=args.sequence_len,
            patch_size=args.feature_num,
            hidden_dim=gat_hidden_dims,
            lstm_hidden_dim=lstm_hidden_dims,
            graph_mode=args.graph_mode,
            embed_dim=args.gat_embed_dim,
            topk=args.gat_topk,
            dropout=args.gat_dropout,
            alpha=args.gat_alpha,
            return_attention=True,
        )

    raise ValueError("--model-structure must be one of ['encoderdecoder', 'original']")

if __name__ == '__main__':
    current_dir = os.getcwd()  # Get the current directory
    parent_dir = os.path.dirname(current_dir)  # Get the upper-level directory
    parser = argparse.ArgumentParser(description='Cmapss Dataset With Pytorch')
    parent_dir = "/CMAPSS-release"
    parser.add_argument('--sequence-len', type=int, default=30)
    parser.add_argument('--feature-num', type=int, default=14)
    parser.add_argument('--dataset-root', type=str,
                        default=parent_dir + '/CMAPSSData/',
                        help='The dir of CMAPSS dataset1')
    parser.add_argument('--sub-dataset', type=str, default='FD001', help='FD001/2/3/4')
    parser.add_argument('--max-rul', type=int, default=125, help='piece-wise RUL')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=2e-3)
    parser.add_argument('--lr-scheduler', type=str, default='step', choices=['step'],
                        help='learning rate scheduler type (step only)')
    parser.add_argument('--step-size', type=int, default=10, help='interval of learning rate scheduler')
    parser.add_argument('--gamma', type=float, default=0.1, help='ratio of learning rate scheduler')
    parser.add_argument('--weight-decay', type=float, default=1e-5)
    parser.add_argument('--patience', type=int, default=8, help='Early Stop Patience')
    parser.add_argument('--max-epochs', type=int, default=30)
    parser.add_argument('--use-exponential-smoothing', default=True)  
    parser.add_argument('--smooth-rate', type=int, default=40)
    parser.add_argument('--use-spatial-gat', dest='use_spatial_gat', action='store_true', default=True,
                        help='Enable sensor graph attention before LSTM encoder')
    parser.add_argument('--disable-spatial-gat', dest='use_spatial_gat', action='store_false',
                        help='Disable sensor graph attention and use raw sensor sequence')
    parser.add_argument('--gat-hidden-dim', type=int, default=8,
                        help='Hidden size for intermediate GAT layers')
    parser.add_argument('--gat-out-dim', type=int, default=16,
                        help='Output embedding size of final GAT layer')
    parser.add_argument('--gat-num-layers', type=int, default=2, choices=[1, 2, 3],
                        help='Number of spatial GAT layers (1/2/3)')
    parser.add_argument('--gat-dropout', type=float, default=0.0,
                        help='Dropout used inside graph attention layers')
    parser.add_argument('--gat-alpha', type=float, default=0.1,
                        help='Negative slope for graph attention LeakyReLU')
    parser.add_argument('--gat-embed-dim', type=int, default=8,
                        help='Sensor node embedding size used to build cosine-similarity graph')
    parser.add_argument('--gat-topk', type=int, default=7,
                        help='Top-k neighbors per sensor selected by cosine similarity')
    parser.add_argument('--graph-mode', type=str, default='dynamic_knn',
                        choices=['dynamic_knn', 'path'],
                        help='Graph construction mode for spatial GAT')
    parser.add_argument('--decoder-attention-size', type=int, default=28,
                        help='Hidden size for decoder additive attention')
    parser.add_argument('--use-aef', dest='use_aef', action='store_true', default=True,
                        help='Enable AEF additive-attention branch in decoder path')
    parser.add_argument('--disable-aef', dest='use_aef', action='store_false',
                        help='Disable AEF and fallback to last encoder step in decoder path')
    parser.add_argument('--use-decoder', dest='use_decoder', action='store_true', default=True,
                        help='Use decoder LSTM head for encoderdecoder structure')
    parser.add_argument('--disable-decoder', dest='use_decoder', action='store_false',help='Disable decoder and use encoder-only regression head')

    parser.add_argument('--apply-code-ablation', dest='apply_code_ablation', action='store_true', default=True,
                        help='Auto-apply A-D ablation preset from --model-code')
    parser.add_argument('--disable-code-ablation', dest='apply_code_ablation', action='store_false',
                        help='Disable auto ablation preset from --model-code')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--save-model', dest='save_model', action='store_true', default=True,
                        help='save trained models')
    parser.add_argument('--no-save-model', dest='save_model', action='store_false',
                        help='do not save trained models')
    parser.add_argument('--model-code', type=str, default='A',
                        help='模型代号，用于日志命名，如 A/B/C/D')
    parser.add_argument('--model-structure', type=str, default='encoderdecoder',
                        choices=['encoderdecoder', 'original'],
                        help='模型结构选择: encoderdecoder / original(=GAT_LSTM_model)')
    parser.add_argument('--lstm-hidden-dim', type=int, default=8,
                        help='Hidden size of LSTM for lstm/original structures')
    parser.add_argument('--seed-list', type=int, nargs='+', default=None,
                        help='Override training seeds, e.g. --seed-list 62 80 88 97')
    parser.add_argument('--start-seed', type=int, default=None,
                        help='Start from this seed within the active seed list')
    args = parser.parse_args()

    model_code = str(args.model_code).strip()
    if not model_code:
        raise ValueError("--model-code cannot be empty.")

    ablation_preset_info = None
    if args.apply_code_ablation:
        ablation_preset_info = _apply_ablation_preset_from_code(args, model_code)
        if ablation_preset_info is not None:
            print(f"[Ablation Preset] model-code={model_code} -> {ablation_preset_info[0]}: {ablation_preset_info[1]}")

    run_output_root = os.path.join(default_log_dir, f"{args.sub_dataset}_{model_code}")
    os.makedirs(run_output_root, exist_ok=True)

    seed_sequence = list(DEFAULT_SEEDS if args.seed_list is None else args.seed_list)
    if not seed_sequence:
        raise ValueError("seed sequence is empty. Use --seed-list with at least one seed.")
    if args.start_seed is not None:
        if args.start_seed not in seed_sequence:
            raise ValueError(f"--start-seed {args.start_seed} is not in active seed list: {seed_sequence}")
        seed_sequence = seed_sequence[seed_sequence.index(args.start_seed):]
    print(f"[Seed Plan] {seed_sequence}")

    for num in seed_sequence:

        torch.manual_seed(num)


        train_loader, valid_loader, test_loader, test_loader_last, \
            num_test_windows, train_visualize, engine_id = get_dataloader(
                dir_path=args.dataset_root,
                sub_dataset=args.sub_dataset,
                max_rul=args.max_rul,
                seq_length=args.sequence_len,
                batch_size=args.batch_size,
                use_exponential_smoothing=args.use_exponential_smoothing,
                smooth_rate=args.smooth_rate)

        model = _build_model_by_structure(args)

        model_type = type(model).__name__

        criterion_train = torch.nn.MSELoss()
        criterion_eval = RMSELoss()
        optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=optimizer,
            step_size=args.step_size,
            gamma=args.gamma,
        )

        seed_output_dir = os.path.join(run_output_root, f"seed{num}")
        os.makedirs(seed_output_dir, exist_ok=True)

        time = datetime.datetime.now().strftime("%m%d_%H%M%S")
        run_tag = f"{time}_seed{num}"
        log_file_name = _build_log_name(
            model_code=model_code,
            lr=args.lr,
            embed_dim=args.gat_embed_dim,
            topk=args.gat_topk,
            time_tag=run_tag,
        )
        log_path = os.path.join(seed_output_dir, log_file_name)

        with open(log_path, "a", encoding="utf-8") as f:

            f.write("-----"+ args.sub_dataset + "-----\n")
            f.write("关键参数:\n")
            f.write(f"序列长度: {args.sequence_len}\n")
            f.write(f"批大小(batch_size): {args.batch_size}\n")
            f.write(f"最大训练轮数(max_epochs): {args.max_epochs}\n")
            f.write(f"随机数种子: {num}\n")
            f.write(f"模型代号(model_code): {args.model_code}\n")
            f.write(f"模型结构(model_structure): {args.model_structure}\n")
            if ablation_preset_info is not None:
                f.write(f"消融预设(ablation_preset): {ablation_preset_info[0]} - {ablation_preset_info[1]}\n")
            f.write(f"学习率: {args.lr}\n")
            f.write("学习率调度器: step\n")
            f.write(f"Step步长(step_size): {args.step_size}\n")
            f.write(f"Step衰减系数(gamma): {args.gamma}\n")
            
            if args.model_structure == 'encoderdecoder':
                f.write(f"空间GAT: {'开启' if args.use_spatial_gat else '关闭'}\n")
                f.write(f"是否使用解码器LSTM: {'是' if args.use_decoder else '否'}\n")
                f.write(f"是否使用AEF: {'是' if args.use_aef else '否'}\n")
            else:
                f.write("空间GAT: 由模型内部默认启用\n")
            if args.model_structure == 'encoderdecoder' and args.use_decoder:
                f.write(f"Decoder注意力隐层维度: {args.decoder_attention_size}\n")
            if args.model_structure != 'encoderdecoder' or args.use_spatial_gat:
                f.write(f"图构建模式(graph_mode): {args.graph_mode}\n")
                f.write(f"GAT层数(num_layers): {args.gat_num_layers}\n")
                f.write(f"GAT邻居数(topk): {args.gat_topk}\n")
                f.write(f"GAT嵌入维度: {args.gat_embed_dim}\n")
                f.write(f"GAT隐藏层维度: {args.gat_hidden_dim}\n")
            f.write("------------------------------\n")


        train(
            model, train_loader, valid_loader,
            test_loader, args.max_epochs, optimizer,
            scheduler, criterion_train, criterion_eval,
            lines_list=[], patience=args.patience, max_rul=args.max_rul, num_test_windows=num_test_windows,
            device=torch.device('cuda') if not args.no_cuda else torch.device('cpu'), time=time, log_path=log_path,
            checkpoint_dir=seed_output_dir,
            checkpoint_prefix='model_' + args.sub_dataset + '_' + run_tag)

        model_output_dir = seed_output_dir
        os.makedirs(model_output_dir, exist_ok=True)

        saved_model_path = os.path.join(model_output_dir, 'model_' + args.sub_dataset + '_' + run_tag + '.pkl')

        if args.save_model:
            torch.save(model, saved_model_path)
        else:
            # train() 会在验证最优时写入 best_*.pkl
            saved_model_path = os.path.join(model_output_dir, 'best_model_' + args.sub_dataset + '_' + run_tag + '.pkl')

        auto_test_cmd = [
            sys.executable,
            os.path.join(parent_dir, 'scipt', 'test_model.py'),
            '--sub-dataset', args.sub_dataset,
            '--smooth-rate', str(args.smooth_rate),
            '--sequence-len', str(args.sequence_len),
            '--feature-num', str(args.feature_num),
            '--dataset-root', args.dataset_root,
            '--max-rul', str(args.max_rul),
            '--batch-size', str(args.batch_size),
            '--model-path', saved_model_path,
        ]
        if args.no_cuda:
            auto_test_cmd.append('--no-cuda')

        test_result = subprocess.run(
            auto_test_cmd,
            cwd=parent_dir,
            capture_output=True,
            text=True,
        )

        with open(log_path, "a", encoding="utf-8") as f:
            f.write("[Auto Test] command: " + " ".join(auto_test_cmd) + "\n")
            if test_result.stdout:
                f.write(test_result.stdout.strip() + "\n")
            if test_result.returncode != 0:
                f.write("[Auto Test][ERROR] return_code={}\n".format(test_result.returncode))
                if test_result.stderr:
                    f.write(test_result.stderr.strip() + "\n")

        if test_result.returncode == 0:
            print("[Auto Test] completed. Result appended to log file.")
            if test_result.stdout:
                print(test_result.stdout.strip())
        else:
            print("[Auto Test][ERROR] Failed to run test_model.py. See log for details.")
