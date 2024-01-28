import json
from typing import Iterator, Optional, Union

import pandas as pd
import torch
from data.kafka_assignor import CustomPartitionAssignor
from kafka import KafkaConsumer
from kafka.coordinator.assignors.range import RangePartitionAssignor
from kafka.coordinator.assignors.roundrobin import RoundRobinPartitionAssignor
from torch.utils.data import IterableDataset
from torch.utils.data.dataset import T_co


class EvenflowDataset(IterableDataset):
    def __init__(self):
        self._mode: str = None  # kafka or local
        self._consumer: Optional[Union[KafkaConsumer, any]] = None
        self.pp_val: torch.Tensor = torch.Tensor([-1.0])
        self.partition_strategy = None

    @classmethod
    def from_kafka(
        cls, world_size: int, rank: int, bootstrap_servers: str, topic: str, partition_strategy_name: str
    ) -> "EvenflowDataset":
        _obj: EvenflowDataset = cls()
        _obj._mode = "kafka"
        _obj.partition_strategy = (
            (RangePartitionAssignor, RoundRobinPartitionAssignor)
            if partition_strategy_name == "standard"
            else [CustomPartitionAssignor(world_size, rank)]
        )
        _obj._consumer = KafkaConsumer(
            topic,
            bootstrap_servers=bootstrap_servers,
            client_id=rank,
            group_id=f"gid_{rank}_{world_size}",
            value_deserializer=bytes.decode,
            partition_assignment_strategy=_obj.partition_strategy,
            enable_auto_commit=False,
            auto_offset_reset="earliest",
            consumer_timeout_ms=100,
        )
        return _obj

    @classmethod
    def from_local_file(cls, filepath: str, chunk_size: int = 1000) -> "EvenflowDataset":
        _obj: EvenflowDataset = cls()
        _obj._mode = "local"
        _obj._consumer = pd.read_csv(filepath, chunksize=chunk_size, on_bad_lines="skip")
        return _obj

    @classmethod
    def from_args(cls, training: bool, **kwargs) -> "EvenflowDataset":
        ds_source: str = kwargs["ds_source"]

        if ds_source == "local":
            file_path: str = kwargs["ds_location"]
            chunk_size: int = kwargs.get("chunk_size", 1000)
            return EvenflowDataset.from_local_file(file_path, chunk_size)

        if ds_source == "kafka":
            world_size: int = kwargs["world_size"]
            rank: int = kwargs["rank"]
            bootstrap_servers: str = kwargs["ds_location"]
            topic: str = kwargs["train_topic"] if training else kwargs["test_topic"]
            partition_strategy: str = kwargs["partition_strategy"]
            return EvenflowDataset.from_kafka(world_size, rank, bootstrap_servers, topic, partition_strategy)

        raise ValueError(f"Unknown dataset source: {ds_source}")

    def _internal_iter(self) -> Iterator[T_co]:
        if self._mode == "kafka":
            while True:
                # Can increase the number of fetched records per call
                kafka_records: dict = self._consumer.poll(timeout_ms=100, max_records=1)
                for _, consumer_records in kafka_records.items():
                    for consumer_record in consumer_records:
                        payload: dict[str, any] = json.loads(consumer_record.value)
                        x: torch.Tensor = torch.as_tensor(payload["values"], dtype=torch.float32)
                        y: torch.Tensor = torch.as_tensor([payload["class"]], dtype=torch.float32)
                        yield x, y
        elif self._mode == "local":
            for chunk in self._consumer:
                for row in chunk.values:
                    x: torch.Tensor = torch.as_tensor(row[1:], dtype=torch.float32)
                    y: torch.Tensor = torch.as_tensor([row[0]], dtype=torch.float32)
                    yield x, y
        else:
            raise ValueError(f"Unknown mode: {self._mode}")

    def __iter__(self) -> Iterator[T_co]:
        remaining_pills: int = 0
        assigned_partitions: bool = False

        for x, y in self._internal_iter():
            # Pull the Kafka assigned partitions from the cluster
            if not assigned_partitions:
                remaining_pills = len(self._consumer.assignment())
                print(f"Number of poison expected poisoned pills: {remaining_pills}")
                assigned_partitions = True

            # Check if a poison pill has been received
            if torch.equal(y, self.pp_val):
                remaining_pills -= 1
                print(f"Received poison pill. Remaining pills: {remaining_pills}")
            elif remaining_pills > 0:
                yield x, y

            if remaining_pills <= 0:
                print("Received all poison pills. Exiting")
                return

    def close(self):
        self._consumer.close()
