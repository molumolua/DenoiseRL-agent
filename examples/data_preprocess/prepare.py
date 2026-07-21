# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Prepare placeholder parquet files for verl-agent environments."""

import os
import datasets

try:
    from verl.utils.hdfs_io import copy, makedirs
except ImportError:
    # Allows running this script without the full verl stack (e.g. on a
    # no-network machine that just needs to (re)generate text parquet files).
    import shutil

    def makedirs(name, mode=0o777, exist_ok=False, **kwargs):
        os.makedirs(name, mode=mode, exist_ok=exist_ok)

    def copy(src, dst, **kwargs):
        if os.path.isdir(src):
            return shutil.copytree(src, dst, **kwargs)
        return shutil.copy(src, dst, **kwargs)

import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='visual', choices=['visual', 'text'])
    parser.add_argument('--local_dir', default='~/data/verl-agent/')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--train_data_size', default=256, type=int)
    parser.add_argument('--val_data_size', default=256, type=int)
    # Visual mode still uses an external image dataset. Text mode is generated
    # locally and never contacts the Hub.
    parser.add_argument(
        '--offline',
        action='store_true',
        default=os.getenv('HF_HUB_OFFLINE', '0') == '1',
        help='Force visual-mode source data to use the local HF cache only.',
    )

    args = parser.parse_args()
    print(f"processing data for mode: {args.mode}")
    args.local_dir = os.path.join(args.local_dir, args.mode)
    os.makedirs(args.local_dir, exist_ok=True)

    if args.offline:
        os.environ.setdefault('HF_HUB_OFFLINE', '1')
        os.environ.setdefault('HF_DATASETS_OFFLINE', '1')

    if args.mode == 'text':
        # Text-mode rows are only placeholders that determine how many ALFWorld
        # environments the trainer creates. The task text and rewards come from
        # ALFWorld at rollout time, so no external dataset is needed here.
        def make_text_dataset(split, size):
            return datasets.Dataset.from_dict({
                'answer': [''] * size,
                'data_source': ['text'] * size,
                'prompt': [[{'role': 'user', 'content': ''}] for _ in range(size)],
                'ability': ['agent'] * size,
                'extra_info': [
                    {'split': split, 'index': idx}
                    for idx in range(size)
                ],
            })

        train_dataset = make_text_dataset('train', args.train_data_size)
        test_dataset = make_text_dataset('test', args.val_data_size)
    else:
        # Visual mode uses Geometry3K only as a source of image-shaped rows.
        # Its problem statements and answers are not used by the agent task.
        dataset = datasets.load_dataset('hiyouga/geometry3k')
        train_dataset = dataset['train'].select(range(args.train_data_size))
        test_dataset = dataset['test'].select(range(args.val_data_size))

        def make_visual_row(example, idx, split):
            return {
                'data_source': 'visual',
                'prompt': [{
                    'role': 'user',
                    'content': '<image>',
                }],
                'images': example['images'],
                'ability': 'agent',
                'extra_info': {
                    'split': split,
                    'index': idx,
                },
            }

        train_dataset = train_dataset.map(
            function=make_visual_row,
            fn_kwargs={'split': 'train'},
            with_indices=True,
            num_proc=8,
            remove_columns=['problem'],
        )
        test_dataset = test_dataset.map(
            function=make_visual_row,
            fn_kwargs={'split': 'test'},
            with_indices=True,
            num_proc=8,
            remove_columns=['problem'],
        )

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)
