import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


def detect_task_from_values(values):
    numeric = True
    cast_values = []
    for v in values:
        if v is None:
            continue
        s = str(v).strip()
        if s == '':
            continue
        try:
            fv = float(s)
            cast_values.append(fv)
        except ValueError:
            numeric = False
            break

    if not numeric:
        return 'multiclass'

    unique_vals = set(cast_values)
    if len(unique_vals) <= 2 and unique_vals.issubset({0.0, 1.0}):
        return 'binclass'

    if all(float(v).is_integer() for v in unique_vals) and 2 < len(unique_vals) <= 30:
        return 'multiclass'

    return 'regression'


def detect_task_from_csv(csv_file: Path):
    with csv_file.open('r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        if not header:
            raise ValueError(f'empty header in {csv_file}')
        label_values = []
        for row in reader:
            if not row:
                continue
            label_values.append(row[-1])
    if not label_values:
        raise ValueError(f'no data rows in {csv_file}')
    return detect_task_from_values(label_values)


def get_metric_key(task):
    return {
        'binclass': 'roc_auc',
        'regression': 'rmse',
        'multiclass': 'accuracy',
    }[task]


def run_cmd(cmd):
    print('>>>', ' '.join(cmd))
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True, help='Directory containing *.csv datasets')
    parser.add_argument('--checkpoint_bin', required=True, help='TP-BERTa checkpoint for binclass/multiclass')
    parser.add_argument('--checkpoint_reg', required=True, help='TP-BERTa checkpoint for regression')
    parser.add_argument('--model_output_dir', required=True, help='Directory to save fine-tuned models')
    parser.add_argument('--result_csv', required=True, help='CSV file for cumulative benchmark results')
    parser.add_argument('--run_result_dir', default='finetune_outputs_batch', help='Intermediate run outputs')
    parser.add_argument('--max_epochs', type=int, default=200)
    parser.add_argument('--early_stop', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[3]
    per_dataset_script = repo_root / 'scripts' / 'finetune' / 'default' / 'run_default_config_tpberta.py'

    data_dir = Path(args.data_dir)
    model_output_dir = Path(args.model_output_dir)
    model_output_dir.mkdir(parents=True, exist_ok=True)

    result_csv = Path(args.result_csv)
    result_csv.parent.mkdir(parents=True, exist_ok=True)

    dataset_files = sorted(data_dir.glob('*.csv'))
    if not dataset_files:
        raise RuntimeError(f'No csv files found in {data_dir}')

    rows = []
    for csv_file in dataset_files:
        dataset_name = csv_file.stem
        task = detect_task_from_csv(csv_file)
        checkpoint_dir = args.checkpoint_reg if task == 'regression' else args.checkpoint_bin
        model_name = f'{dataset_name}_tpberta'

        run_cmd([
            sys.executable,
            str(per_dataset_script),
            '--dataset', dataset_name,
            '--task', task,
            '--data_dir', str(data_dir),
            '--checkpoint_dir', str(checkpoint_dir),
            '--result_dir', args.run_result_dir,
            '--model_output_dir', str(model_output_dir),
            '--model_name', model_name,
            '--max_epochs', str(args.max_epochs),
            '--early_stop', str(args.early_stop),
            '--batch_size', str(args.batch_size),
            '--lr', str(args.lr),
            '--weight_decay', str(args.weight_decay),
        ])

        finish_json = Path(args.run_result_dir) / task / 'TPBerta-default' / dataset_name / 'finish.json'
        with finish_json.open('r', encoding='utf-8') as f:
            finish = json.load(f)
        metric_key = get_metric_key(task)
        metric = finish['final_test_score']

        finetune_args = finish.get('args', {})

        rows.append({
            'dataset_name': dataset_name,
            'model_name': 'tp-berta',
            'metric': metric,
            'metric_name': metric_key,
            'task': task,
            'saved_model_file': str((model_output_dir / f'{model_name}.pt').resolve()),
            'lr': finetune_args.get('lr', args.lr),
            'weight_decay': finetune_args.get('weight_decay', args.weight_decay),
            'batch_size': finetune_args.get('batch_size', args.batch_size),
            'max_epochs': finetune_args.get('max_epochs', args.max_epochs),
            'early_stop': finetune_args.get('early_stop', args.early_stop),
            'checkpoint_dir': str(checkpoint_dir),
            'finetune_params_json': json.dumps(finetune_args, ensure_ascii=False),
        })

        print(f'[DONE] dataset={dataset_name}, task={task}, metric({metric_key})={metric}')

    write_header = not result_csv.exists() or result_csv.stat().st_size == 0
    with result_csv.open('a', newline='', encoding='utf-8') as f:
        fieldnames = [
            'dataset_name', 'model_name', 'metric', 'metric_name', 'task', 'saved_model_file',
            'lr', 'weight_decay', 'batch_size', 'max_epochs', 'early_stop', 'checkpoint_dir', 'finetune_params_json'
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)

    print(f'Appended {len(rows)} records to {result_csv}')


if __name__ == '__main__':
    main()
