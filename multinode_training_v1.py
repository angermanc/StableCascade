import ray

import socket
import subprocess
import multiprocessing
from typing import Optional
from typing import Any, Generic, Optional, Type, TypeVar

import time

T_store = TypeVar("T_store")


class RayObjectStoreType(Generic[T_store]):
    def __init__(self) -> None:
        self.__lock = multiprocessing.Lock()
        self.__value: Optional[T_store] = None

    def set(self, value: T_store) -> None:
        with self.__lock:
            self.__value = value

    def clear(self) -> None:
        with self.__lock:
            self.__value = None

    def get(self) -> Optional[T_store]:
        with self.__lock:
            return self.__value


@ray.remote
class RayObjectStore(RayObjectStoreType):
    pass


def _get_object_store_actor(
    name: str, type: Type[T_store], get_only: bool = False
) -> RayObjectStoreType[T_store]:
    if get_only:
        return ray.get_actor(name=name, namespace="torchrun")
    else:
        return RayObjectStore.options(
            name=name,
            namespace="torchrun",
            lifetime="detached",
            max_restarts=0,
            get_if_exists=True,
        ).remote()


def init_torchrun(
    row: dict[str, any],
    nnodes_with_gpu: int,
    num_gpu_per_node: int,
    object_store_handle: Any,
) -> None:
    rank = row["id"]
    if rank == 0:
        master_address = socket.gethostbyname(socket.gethostname())
        master_port = 29400
        object_store_handle.set.remote((master_address, master_port))
    c10d_master_values = None
    while c10d_master_values is None:
        c10d_master_values = object_store_handle.get.remote()
        print("Waiting on c10d master connection info")
        time.sleep(1)
    print(f"c10d master values {c10d_master_values}")
    
    master_addr, master_port = ray.get(c10d_master_values)

    cmd = (
        f"torchrun "
        f"--nnodes {nnodes_with_gpu} "
        f"--node_rank {rank} "
        f"--nproc_per_node {num_gpu_per_node} "
        f"--master_addr {master_addr} "
        f"--master_port {master_port} "
        "train/train_c.py configs/training/finetune_c_3b.yaml"
    )
    print(f'CMD {cmd}')
    subprocess.run(cmd, shell=True, check=True)
    return {"success": 1}


def main() -> None:
    ray.init()
    ray.data.DataContext.get_current().max_errored_blocks = -1


    nnodes_with_gpu = 2
    num_gpu_per_node = 8
    # https://aws.amazon.com/ec2/instance-types/p4/
    num_cpus = 96  # set this to num cpus on the worker node, so to make sure that each ray task gets scheduled on a separate worker node

    object_store_handle = _get_object_store_actor(
        "c10d_store", type=tuple, get_only=False
    )

    ds = ray.data.range(nnodes_with_gpu).map(
        init_torchrun,
        fn_kwargs={
            "nnodes_with_gpu": nnodes_with_gpu,
            "num_gpu_per_node": num_gpu_per_node,
            "object_store_handle": object_store_handle,
        },
        num_gpus=num_gpu_per_node,
        concurrency=nnodes_with_gpu,
        # num_cpus=num_cpus,
    )
    ds.materialize()


if __name__ == "__main__":
    main()