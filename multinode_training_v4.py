import ray
from ray.util.placement_group import placement_group, remove_placement_group
import socket
import subprocess
import multiprocessing
from typing import Optional
from typing import Any, Generic, Optional, Type, TypeVar

import psutil

import time
import pyarrow.fs
from data_prep.utils import configure_aws_assume_role_provider


MODEL_TRAINER_ARN = (
    "arn:aws:iam::051687089423:role/service.ingredient-generation-model-trainer"
)


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


NUM_GPU_PER_NODE = 8


@ray.remote(num_gpus=NUM_GPU_PER_NODE)
def init_torchrun(
    rank: int,
    nnodes_with_gpu: int,
    num_gpu_per_node: int,
    object_store_handle: Any,
    train_script_cmd: str,
    train_script_config: str,
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
        f"{train_script_cmd} {train_script_config}"
    )
    subprocess.run(cmd, shell=True, check=True)
    return {"success": 1}


def main() -> None:
    ray.init()
    nnodes_with_gpu = 2
    train_script_cmd = "-m train.train_c"
    train_script_config = "configs/training/finetune_c_3b.yaml"

    object_store_handle = _get_object_store_actor(
        "c10d_store", type=tuple, get_only=False
    )
    tasks = []
    for i in range(nnodes_with_gpu):
        # we get a future task ref
        task_ref = init_torchrun.remote(
            rank=i,
            nnodes_with_gpu=nnodes_with_gpu,
            num_gpu_per_node=NUM_GPU_PER_NODE,
            object_store_handle=object_store_handle,
            train_script_cmd=train_script_cmd,
            train_script_config=train_script_config,
        )
        tasks.append(task_ref)

    try:
        for task in tasks:
            # ray.get blocks until task finishes
            ray.get(task)
    except KeyboardInterrupt:
        cleanup_torchrun_tasks()
    finally:
        # Cleanup detached actor
        ray.kill(object_store_handle)


def cleanup_torchrun_tasks():
    """Launches task to SIGTERM torchrun on the worker nodes"""
    num_nodes = len(ray.nodes())
    bundles = [{"CPU": 1} for _ in range(num_nodes)]
    pg = placement_group(bundles=bundles, strategy="STRICT_SPREAD")
    ray.get(pg.ready())

    # Schedule kill_process_by_name tasks on each node
    kill_tasks = []
    for i in range(num_nodes):
        kill_task = kill_process_by_name.options(
            placement_group=pg, placement_group_bundle_index=i
        ).remote("--master_addr")  # identify processes by torchrun argument
        kill_tasks.append(kill_task)

    # Wait for all kill tasks to complete
    ray.get(kill_tasks)
    remove_placement_group(pg)


@ray.remote
def kill_process_by_name(script_name):
    """
    Searches for and kills all processes running a script with the given name.

    Args:
        script_name (str): The name of the script (e.g., 'main.py')
    """
    for process in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            # Check if the script name is in the command line arguments
            if script_name in process.info["cmdline"]:
                process.terminate()  # Gracefully kill the process
                process.wait(timeout=5)  # Give it some time to terminate
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess) as e:
            print(f"Could not terminate process {process.info['pid']}: {e}")


if __name__ == "__main__":
    fs = pyarrow.fs.S3FileSystem(role_arn=MODEL_TRAINER_ARN, region="us-east-1")
    configure_aws_assume_role_provider(MODEL_TRAINER_ARN)
    main()