import argparse
import copy
import functools
import os
from argparse import ArgumentParser
from typing import Any

import torch
import torch.distributed.rpc as rpc
import torch.multiprocessing as mp
import torch.nn as nn
from aggregate_logs_and_metrics import EvenFlowLogsAndMetricsAgg
from client.evenflow_worker import EvenflowWorker
from data.evenflow_dataset import EvenflowDataset
from evenflow_factory import EvenflowFactory
from server.evenflow_server import EvenflowServer


def spawn_worker(p_args: dict[str, Any], rank: int) -> None:
    p_args["rank"] = rank
    train_dataset: EvenflowDataset = EvenflowDataset.from_args(training=True, **p_args)
    mode: str = p_args.pop("mode")
    trainer: EvenflowWorker = EvenflowFactory.instantiate_client(mode=mode, dataset=train_dataset, **p_args)

    trainer.setup()

    for cur_round in range(1, p_args["total_rounds"] + 1):
        trainer.on_round_start(cur_round)

        # Server provides model and other relevant information to the workers
        round_context: dict[str, any] = p_args["ps_rref"].rpc_sync().provide_round_context_to_worker()
        # Trainer ops
        trainer.accept_round_countext(round_context)
        trainer.train_round(cur_round)
        trainer.on_round_completed(cur_round)

        # Signal server that the round has been completed
        p_args["ps_rref"].rpc_sync().complete_round()

    trainer.teardown()


def runner(rank: int, *args, **kwargs) -> None:
    """Process task"""
    p_args: dict[str, any] = copy.deepcopy(kwargs["p_args"])
    p_args["rank"] = rank

    options: rpc.TensorPipeRpcBackendOptions = rpc.TensorPipeRpcBackendOptions(
        num_worker_threads=p_args["num_worker_threads"], rpc_timeout=p_args["rpc_timeout"]
    )

    # Start RPC connections
    rpc.init_rpc(
        "ps" if rank == 0 else f"trainer{p_args['rank']}",
        rank=p_args["rank"],
        world_size=p_args["world_size"],
        rpc_backend_options=options,
    )

    # Init PS and tasks
    if p_args["rank"] == 0:
        # Create a dataset for training and testing
        test_dataset: EvenflowDataset = EvenflowDataset.from_args(training=False, **p_args)

        # Instantiate the PS
        ps: EvenflowServer = EvenflowFactory.instanciate_server(dataset=test_dataset, **p_args)
        ps.setup()

        # Create RRef of the PS, will be fed to workers
        ps_rref: rpc.RRef[EvenflowServer] = rpc.RRef(ps)
        p_args["ps_rref"] = ps_rref

        # Create and distribute worker tasks
        tasks: list[torch.futures.Future] = [
            rpc.rpc_async(f"trainer{rank}", functools.partial(spawn_worker, rank=rank), args=(p_args,))
            for rank in range(1, p_args["world_size"])
        ]

        # Wait for all worker tasks to complete (will be done after all rounds are over)
        torch.futures.wait_all(tasks)

        # Destroy the PS
        ps.teardown()

    # Wait for tasks to complete
    rpc.shutdown()


if __name__ == "__main__":
    parser: ArgumentParser = argparse.ArgumentParser("Evenflow distributed parameter server")

    parser.add_argument("--num_training_batches", default=3, type=int, help="Batches per training epoch.")
    parser.add_argument("--num_test_batches", default=3, type=int, help="Batches per test epoch.")
    parser.add_argument("--batch_size", default=16, type=int, help="Kafka messages per batch.")
    parser.add_argument("--lr", default=0.001, type=float, help="ADAM learning rate.")
    parser.add_argument("--num_worker_threads", default=32, type=int, help="Number of RPC worker threads.")
    parser.add_argument("--rpc_timeout", default=0, type=int, help="RPC timeout.")
    parser.add_argument("--beta1", default=0.9, type=float, help="ADAM beta1 hyperparameter.")
    parser.add_argument("--beta2", default=0.999, type=float, help="ADAM beta2 hyperparameter.")
    parser.add_argument("--partition_strategy", default="custom", type=str, help="Kafka partition strategy.")
    parser.add_argument(
        "--ds_source", choices=["kafka", "local"], default="kafka", type=str, help="Supported dataset ingestion method."
    )
    parser.add_argument(
        "--ds_location",
        default="localhost:9092",  # Can also be a local file name
        type=str,
        help="Supported ds locations (can be a local dataset file name or a kafka cluster IP).",
    )
    parser.add_argument(
        "--train_topic",
        type=str,
        help="Name of the training topic at --ds_location",
    )
    parser.add_argument(
        "--test_topic",
        type=str,
        help="Name of the training topic at --ds_location",
    )
    parser.add_argument(
        "--mode",
        choices=["sync", "async", "fedopt", "fedoptadv"],
        default="sync",
        type=str,
        help="Distributed training algorithm.",
    )
    parser.add_argument("--total_rounds", default=10, type=int, help="Number of training rounds.")
    parser.add_argument("--world_size", default=6, type=int, help="Number of workers + 1 (PS).")
    parser.add_argument("--master_address", default="localhost", type=str, help="Master node IP address.")
    parser.add_argument("--master_port", default=29500, type=int, help="Master node port.")
    parser.add_argument("--log_root", default="logs/", type=str, help="Root directory for logs.")
    parser.add_argument("--metrics_root", default="metrics/", type=str, help="Root directory for metrics.")
    parser.add_argument(
        "--gm_threshold", default=12, type=float, help="Geometrics monitoring algorithm threshold value."
    )
    parser.add_argument("--use_cuda", action="store_true", default=False, help="Force the use CUDA devices.")
    parser.add_argument("--fft_coeffs", default=3, type=int, help="Number of FFT coefficients to use during FedOpt.")
    parser.add_argument(
        "--max_min_on_sphere_impl",
        type=str,
        default="torch_standard",
        help="Search space exploration method used during FedOpt.",
    )

    parsed_args = parser.parse_args()
    p_args: dict[str, Any] = vars(parsed_args)
    print(f"Using arguments: {p_args}")

    os.environ["MASTER_ADDR"] = p_args["master_address"]
    os.environ["MASTER_PORT"] = str(p_args["master_port"])

    # Force cuda on ALL tensors
    if p_args["use_cuda"]:
        print("Forcing use of CUDA devices.")
        cuda: bool = torch.cuda.is_available()
        if cuda:
            torch.set_default_tensor_type("torch.cuda.FloatTensor")

    # Loss function used among workers and PS
    p_args["loss_fn"] = nn.MSELoss()

    # Create relevant directories
    os.makedirs(p_args["log_root"], exist_ok=True)
    os.makedirs(p_args["metrics_root"], exist_ok=True)

    # Execute
    mp.spawn(
        functools.partial(runner, p_args=p_args), args=(p_args["world_size"],), nprocs=p_args["world_size"], join=True
    )

    # Logs and metrics aggregation (post-processing)
    agg_util: EvenFlowLogsAndMetricsAgg = EvenFlowLogsAndMetricsAgg(
        p_args["metrics_root"], p_args["log_root"], p_args["world_size"]
    )

    try:
        agg_util.aggregate_logs()
        agg_util.aggregate_metrics()
    except BaseException as be:
        print(f"Failed to aggregate logs: {str(be)}")

    print("DONE!")
