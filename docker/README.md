
# Docker environments

## Quickstart

Build a docker image:

```sh
./manager build <cpu/cu111/...>
```

Run an image (mounts apps & data)

```sh
./manager run <cpu/cu111/...>
```

List available images:

```sh
./manager list
```

## Supported versions

- `cpu`: CPU-only version
- `cu111`: CUDA 11.1 (supports NVIDIA GeForce RTX 3090)
