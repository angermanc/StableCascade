import ray

import socket

import threading
import time


def init_torchrun(
    master_addr: str,
    master_port: int,
    rank: int,
    nnodes_with_gpu: int,
    num_gpus_per_node: int,
) -> None:
    import torch.distributed as dist
    import runpy

    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://{master_addr}:{master_port}",
        world_size=nnodes_with_gpu * num_gpus_per_node,
        rank=rank,
    )
    print("Master process initialized the process group.")
    dist.barrier()  # Synchronize processes
    if rank != 0:
        runpy.run_path("StableCascade/train/train_c.py configs/training/finetune_c_1b.yaml", run_name="__main__")
    while True:
        time.sleep(1)


def main() -> None:
    ray.init()

    nnodes_with_gpu = 2
    num_gpu_per_node = 8

    master_address = socket.gethostbyname(socket.gethostname())
    master_port = 29400

    comm_backend_thread = threading.Thread(
        target=init_torchrun,
        args=(master_address, master_port, 0, nnodes_with_gpu, num_gpu_per_node),
    )
    # comm_backend_thread.start()
    remote_init_torchrun = ray.remote(init_torchrun, num_gpus=1)
    obj_ref = remote_init_torchrun.remote(
        rank=0,
        nnodes_with_gpu=2,
        num_gpus_per_node=num_gpu_per_node,
        master_addr=master_address,
        master_port=master_port,
    )

    refs = []
    for i in range(1, nnodes_with_gpu + 1):
        # function remote handle
        remote_init_torchrun = ray.remote(init_torchrun, num_gpus=8)
        obj_ref = remote_init_torchrun.remote(
            rank=i,
            nnodes_with_gpu=2,
            num_gpus_per_node=num_gpu_per_node,
            master_addr=master_address,
            master_port=master_port,
        )
        refs.append(obj_ref)

    try:
        for ref in refs:
            # ray.get blocks until task finishes
            ray.get(ref)
    except KeyboardInterrupt:
        # There is no nice way on terminating a subprocess
        # we try to cancel the whole task
        for ref in refs:
            # Set force to True if it does not gracefully shutdown
            ray.cancel(ref, force=False)
            ray.get(ref)

    comm_backend_thread.join()


if __name__ == "__main__":
    main()