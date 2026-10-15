#!/usr/bin/env python
"""按生成 suite 调度比较；未选择任务始终保留 not_run，整体不得伪称 passed。"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from data_provider.spatial_temporal.config import load_config, safe_name
from data_provider.spatial_temporal.dataset import EpisodePackage, contained_path
from data_provider.spatial_temporal.schema import read_json
from exp.exp_gslaformer_forecasting import _json
from exp.gslaformer_comparison import run_comparison
from exp.gslaformer_inference import evaluate_checkpoint


def run_requests(config, output, requests):
    """同一最终 checkpoint 逐请求重载；哈希不变证明未原地改写权重/统计。"""
    checkpoint = output / 'full_train/checkpoints/best.pt'
    before = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    results = {}
    for name, request in requests.items():
        safe_name(name, 'request_name')
        destination = output / f'forecast_{name}'
        evaluate_checkpoint(config, checkpoint, destination, mode='forecast', request=request)
        if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != before:
            raise AssertionError('forecast 修改了来源 checkpoint')
        metadata = read_json(destination / 'metadata.json')
        results[name] = {'status': 'passed', 'path': str(destination), 'checkpoint_sha256': before,
                         'requests': metadata['requests']}
    _json(output / 'forecast_requests.json', results)
    return results


def verify_m5(fixtures):
    suite = read_json(fixtures / 'mechanisms/M5/suite.json')
    config = load_config(fixtures / 'base/config.json')
    config['data'].update(groups=suite['schema']['groups'], relation_types=suite['schema']['relation_types'], known_future=[])
    bad_root = fixtures / 'mechanisms/M5/bad_inputs'
    EpisodePackage(bad_root / '_valid', config).sample(23, supervised=False)
    results = {}
    for name, expected in suite['fixtures'].items():
        try:
            package = EpisodePackage(contained_path(bad_root, name), config)
            if expected.get('trigger') == 'sample':
                package.sample(23, supervised=False)
        except ValueError as exc:
            pattern = expected['error']
            if name == 'path_escape':
                # files 白名单在路径解析前拒绝该 ../escape 字段，同样守住越界边界。
                pattern = r'越界|unknown=.*\.\./escape'
            if not re.search(pattern, str(exc)):
                raise AssertionError(f'{name} 被非预期错误拦截: {exc}') from exc
            results[name] = {'status': 'passed', 'error': str(exc)}
        else:
            raise AssertionError(f'{name} 未拒绝坏输入')
    return {'status': 'passed', 'scope': 'packaged_bad_inputs_only', 'fixtures': results}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--suite_manifest', required=True, type=Path)
    parser.add_argument('--run_prefix', required=True)
    parser.add_argument('--tasks', nargs='+')
    args = parser.parse_args()
    safe_name(args.run_prefix, 'run_prefix')
    manifest = read_json(args.suite_manifest)
    fixtures = args.suite_manifest.resolve().parent.parent
    tasks = {entry['id']: entry for entry in manifest['tasks']}
    selected = list(tasks) if args.tasks is None else args.tasks
    if set(selected) - tasks.keys() or len(selected) != len(set(selected)):
        parser.error('tasks 必须是不重复的已声明任务')
    output = ROOT / 'results/gslaformer' / args.run_prefix
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    result = {'scope': 'comparison_execution_and_packaged_bad_inputs; not complete model acceptance',
              'command': sys.argv, 'status': 'running',
              'tasks': {task: {'status': 'not_run'} for task in tasks}}
    _json(output / 'suite_result.json', result)
    for task in selected:
        try:
            entry = tasks[task]
            if task == 'M5':
                result['tasks'][task] = verify_m5(fixtures)
            elif task == 'aidc':
                profiles = {}
                for profile, path in entry['profiles'].items():
                    config = load_config(contained_path(fixtures, path))
                    profiles[profile] = run_comparison(config, output / profile)
                    profiles[profile]['forecasts'] = run_requests(config, output / profile,
                        {name: contained_path(fixtures, value) for name, value in entry['requests'].items()})
                result['tasks'][task] = {'status': 'passed' if all(p['status'] == 'passed' for p in profiles.values()) else 'failed', 'profiles': profiles}
            else:
                config = load_config(contained_path(fixtures, entry['config']))
                result['tasks'][task] = run_comparison(config, output / task)
                if task == 'M4':
                    result['tasks'][task]['forecasts'] = run_requests(config, output / task,
                        {'origin_activation': contained_path(fixtures, 'mechanisms/M4/request_origin_activation')})
        except Exception as exc:
            result['tasks'][task] = {'status': 'failed', 'error': str(exc)}
        _json(output / 'suite_result.json', result)
        print(task, result['tasks'][task]['status'], flush=True)
    statuses = [value['status'] for value in result['tasks'].values()]
    result['status'] = 'failed' if 'failed' in statuses else 'partial' if 'not_run' in statuses else 'passed'
    _json(output / 'suite_result.json', result)
    print('suite:', result['status'], flush=True)
    raise SystemExit(1 if result['status'] == 'failed' else 0)


if __name__ == '__main__':
    main()
