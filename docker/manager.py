#!/usr/bin/env python3

import argparse
import os
from pathlib import Path
import subprocess


VERSIONS = [
    ('cpu', 'CPU only'),
    ('cu111', 'CUDA 11.1')
]


def has_docker() -> bool:
    p = subprocess.run(
        ['docker', '--version'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    return p.returncode == 0


def ensure_docker():
    if not has_docker():
        print('Error: Collagen manager requires docker')
        exit(-1)


def ensure_tag(tag):
    tags = [x[0] for x in VERSIONS]
    if tag not in tags:
        print(f'Error: {tag} not found')
        print(f'Available versions: {repr(tags)}')
        exit(-1)


def get_root() -> Path:
    return Path(os.path.realpath(__file__)).absolute().parent.parent


def build_image(path, tag, dockerfile):
    subprocess.run([
        'docker', 'build',
        path,
        '-f', dockerfile,
        '-t', tag
    ])


def run_image(tag, extra='', cmd='/bin/bash', mounts={}, workdir='/home'):
    vmount = []
    for host in mounts:
        vmount += ['-v', f'{host}:{mounts[host]}']
    
    subprocess.run([
        'docker', 'run',
        *vmount,
        *[x for x in extra.split(' ') if x != ''],
        '-w', workdir,
        '--rm', '-it',
        tag,
        *cmd.split(' ')
    ])


def cmd_list(args):
    for tag, help in VERSIONS:
        print(f'{tag}: {help}')


def cmd_build(args):
    ensure_tag(args.name)
    root = get_root()
    print(root)

    build_image(
        path=str(root), 
        tag=f'collagen_{args.name}',
        dockerfile=str(root / 'docker' / f'Dockerfile.{args.name}')
    )


def cmd_run(args):
    ensure_tag(args.name)
    root = get_root()
    
    mounts = {}
    if args.apps is not None:
        mounts[str((root / args.apps).absolute())] = '/mnt/apps'
    if args.data is not None:
        mounts[str((root / args.data).absolute())] = '/mnt/data'
    if args.checkpoints is not None:
        mounts[str((root / args.checkpoints).absolute())] = '/mnt/checkpoints'

    run_image(
        tag=f'collagen_{args.name}',
        extra=args.extra,
        mounts=mounts,
        workdir='/mnt/apps',
    )


def cmd_test(args):
    ensure_tag(args.name)

    run_image(
        tag=f'collagen_{args.name}',
        workdir='/home/',
        cmd='python3 -m unittest discover collagen.test',
    )


if __name__=='__main__':
    ensure_docker()

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command', help='Command to run.')

    p_list = subparsers.add_parser('list')

    p_build = subparsers.add_parser('build')
    p_build.add_argument('name', help='Version name.')

    p_run = subparsers.add_parser('run')
    p_run.add_argument('name', help='Version name.')
    p_run.add_argument('--apps', default='./apps', help='Path to apps folder, mounted at /mnt/apps')
    p_run.add_argument('--data', default=None, help='Path to data folder, mounted at /mnt/data')
    p_run.add_argument('--checkpoints', default=None, help='Path to checkpoints folder, mounted at /mnt/checkpoints')
    p_run.add_argument('--extra', required=False, default='', help='Extra arguments passed directly to `docker run`.')

    p_test = subparsers.add_parser('test')
    p_test.add_argument('name', help='Version name.')

    args = parser.parse_args()

    if args.command == 'list':
        cmd_list(args)
    elif args.command == 'build':
        cmd_build(args)
    elif args.command == 'run':
        cmd_run(args)
    elif args.command == 'test':
        cmd_test(args)
    else:
        parser.print_usage()
