from client.async_worker import AsyncWorker
from client.evenflow_worker import EvenflowWorker
from client.fed_opt_adv_worker import FedOptAdvancedWorker
from client.fed_opt_worker import FedOptWorker
from client.sync_worker import SyncWorker
from data.evenflow_dataset import EvenflowDataset
from server.async_server import AsyncServer
from server.evenflow_server import EvenflowServer
from server.fed_opt_server import FedOptServer
from server.sync_server import SyncServer


class EvenflowFactory:
    @staticmethod
    def instantiate_client(mode: str, dataset: EvenflowDataset, **kwargs) -> EvenflowWorker:
        if mode == "sync":
            return SyncWorker(dataset=dataset, **kwargs)
        if mode == "async":
            return AsyncWorker(dataset=dataset, **kwargs)
        if mode == "fedopt":
            return FedOptWorker(dataset=dataset, **kwargs)
        if mode == "fedoptadv":
            return FedOptAdvancedWorker(dataset=dataset, **kwargs)

        raise ValueError(f"Unknown mode {mode}.")

    @staticmethod
    def instanciate_server(mode: str, **kwargs) -> EvenflowServer:
        if mode == "sync":
            return SyncServer(**kwargs)

        if mode == "async":
            return AsyncServer(**kwargs)

        if mode == "fedopt":
            return FedOptServer(**kwargs)

        if mode == "fedoptadv":
            return FedOptServer(**kwargs)

        raise NotImplementedError(f"Mode {mode} not recognised.")
