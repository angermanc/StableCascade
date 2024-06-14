import ray
from ray.util.placement_group import placement_group
import socket
import subprocess
import multiprocessing
from typing import Optional
from typing import Any, Generic, Optional, Type, TypeVar

import psutil
import os

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


@ray.remote(num_gpus=8)
def init_torchrun(
    rank: int,
    nnodes_with_gpu: int,
    num_gpu_per_node: int,
    object_store_handle: Any,
) -> None:
    if rank == 0:
        master_address = socket.gethostbyname(socket.gethostname())
        master_port = 29400
        object_store_handle.set.remote((master_address, master_port))
    c10d_master_values = None
    while c10d_master_values is None:
        c10d_master_values = object_store_handle.get.remote()
        print("Waiting on c10d master connection info")
        time.sleep(1)
    master_addr, master_port = ray.get(c10d_master_values)
    cmd = (
        f"torchrun "
        f"--nnodes {nnodes_with_gpu} "
        f"--node_rank {rank} "
        f"--nproc_per_node {num_gpu_per_node} "
        f"--master_addr {master_addr} "
        f"--master_port {master_port} "
        "-m train.train_c configs/training/finetune_c_1b.yaml"
    )
    subprocess.run(cmd, shell=True, check=True)
    return {"success": 1}


def main() -> None:
    ray.init()
    ray.data.DataContext.get_current().max_errored_blocks = -1


    nnodes_with_gpu = 2
    num_gpu_per_node = 8

    object_store_handle = _get_object_store_actor(
        "c10d_store", type=tuple, get_only=False
    )
    tasks = []
    for i in range(nnodes_with_gpu):
        # we get a future task ref
        task_ref = init_torchrun.remote(
            rank=i,
            nnodes_with_gpu=2,
            num_gpu_per_node=num_gpu_per_node,
            object_store_handle=object_store_handle,
        )
        tasks.append(task_ref)

    try:
        for task in tasks:
            # ray.get blocks until task finishes
            ray.get(task)
    except KeyboardInterrupt:
        # There is no nice way on terminating a subprocess
        # we try to cancel the whole task
        for task in tasks:
            # Set force to True if it does not gracefully shutdown
            ray.cancel(task, force=True)
            # ray.get(task)
        num_nodes = len(ray.nodes())
        bundles = [{"CPU": 1} for _ in range(num_nodes)]
        pg = placement_group(bundles=bundles, strategy="STRICT_SPREAD")
        ray.get(pg.ready())
        # executors = [FunctionExecutor.options(placement_group=pg).remote() for _ in range(num_nodes)]
        # ips = ray.get([kill_process_by_name.remote('train.train_c') for executor in executors])
        for _ in range(num_nodes):
            task = kill_process_by_name.options(placement_group=pg).remote("train.train_c")
            ray.get(task)
        # for _ in range(nnodes_with_gpu):
        #     kill_task = kill_process_by_name.remote(
                
        #     )
            # ray.get(kill_task)
        print('INTERRUPTION SUCCESSFUL')
    # Cleanup detached actor
    ray.kill(object_store_handle)

@ray.remote
def kill_process_by_name(script_name):
    """
    Searches for and kills all processes running a script with the given name.

    Args:
        script_name (str): The name of the script (e.g., 'main.py')
    """
    for process in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            # Check if the script name is in the command line arguments
            if script_name in process.info['cmdline']:
                print(f"Killing process {process.info['pid']} running {script_name}")
                process.terminate()  # Gracefully kill the process
                process.wait(timeout=5)  # Give it some time to terminate
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess) as e:
            print(f"Could not terminate process {process.info['pid']}: {e}")



if __name__ == "__main__":
    main()