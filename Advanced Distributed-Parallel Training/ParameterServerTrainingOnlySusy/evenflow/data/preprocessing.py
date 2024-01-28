import argparse
import json
from argparse import ArgumentParser
from time import sleep, time
from typing import Any

import pandas as pd
from kafka import KafkaAdminClient, KafkaProducer
from kafka.admin import NewTopic
from kafka.errors import KafkaError, UnknownTopicOrPartitionError
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    parser: ArgumentParser = argparse.ArgumentParser("Evenflow Kafka topic setup utility.")
    parser.add_argument("--chunk_size", default=1024, type=int, help="Number of messages to be loaded per file chunk.")
    parser.add_argument("--input_file_path", type=str, help="Input file path.")
    parser.add_argument("--train_topic", type=str, help="Topic with training data.")
    parser.add_argument("--test_topic", type=str, help="Topic with test data.")
    parser.add_argument("--num_chunks", type=int, default=1, help="Number of chunks")
    parser.add_argument("--num_partitions", type=int, default=4, help="Number of Kafka topic partitions.")
    parser.add_argument("--train_test_ratio", type=float, default=0.8, help="Train/Test split, e.g. 0.8")
    parser.add_argument(
        "--bootstrap_servers",
        type=str,
        default="localhost:9092",
        help="Bootstrap servers, currently only one supported",
    )

    parsed_args = parser.parse_args()
    p_args: dict[str, Any] = vars(parsed_args)
    print(f"Using arguments: {p_args}")

    max_records_to_load: int = p_args["chunk_size"] * p_args["num_chunks"]
    bootstrap_servers: list = [p_args["bootstrap_servers"]]

    COLUMNS = [
        #  labels
        "class",
        #  low-level features
        "lepton_1_pT",
        "lepton_1_eta",
        "lepton_1_phi",
        "lepton_2_pT",
        "lepton_2_eta",
        "lepton_2_phi",
        "missing_energy_magnitude",
        "missing_energy_phi",
        #  high-level derived features
        "MET_rel",
        "axial_MET",
        "M_R",
        "M_TR_2",
        "R",
        "MT2",
        "S_R",
        "M_Delta_R",
        "dPhi_r_b",
        "cos(theta_r1)",
    ]

    producer: KafkaProducer = KafkaProducer(bootstrap_servers=bootstrap_servers)
    admin_client: KafkaAdminClient = KafkaAdminClient(bootstrap_servers=bootstrap_servers)

    # Delete existing topics with same name
    try:
        del_resp = admin_client.delete_topics(topics=[p_args["train_topic"], p_args["test_topic"]])
        print(del_resp)
        print("Waiting 5 seconds.")
        sleep(5)
    except UnknownTopicOrPartitionError as _:  # noqa: F841
        pass

    # Create and configure topics
    try:
        create_resp = admin_client.create_topics(
            new_topics=[
                NewTopic(name=p_args["train_topic"], num_partitions=p_args["num_partitions"], replication_factor=1),
                NewTopic(name=p_args["test_topic"], num_partitions=p_args["num_partitions"], replication_factor=1),
            ],
            validate_only=False,
        )
        print(create_resp)
    except KafkaError as _:  # noqa: F841
        pass

    # Load data
    ds_source: pd.DataFrame = pd.read_csv(
        p_args["input_file_path"], header=None, names=COLUMNS, chunksize=p_args["chunk_size"]
    )

    def error_callback(exc):
        raise RuntimeError(f"Error while sending data to kafka: {str(exc)}")

    def write_to_kafka(topic_name, items):
        count = 0
        for message, key in items:
            cur_ts: int = int(time())
            msg_val: bytes = json.dumps(
                {
                    "timestamp": cur_ts,
                    "class": int(float(key)),
                    "values": list(map(float, message.split(","))),
                }
            ).encode("utf-8")

            partition: int = count % p_args["num_partitions"]

            # Send the message to the Kafka broker
            producer.send(topic_name, value=msg_val, partition=partition).add_errback(error_callback)
            count += 1

        # Queue poison pills, one per partition
        # Consumers will break their poll loop once they encounter a poison pill
        for p in range(p_args["num_partitions"]):
            pill: bytes = json.dumps(
                {
                    "timestamp": cur_ts,
                    "class": -1.0,
                    "values": [-1.0 for _ in range(18)],
                }
            ).encode("utf-8")

            producer.send(
                topic_name,
                value=pill,
                partition=p,
            ).add_errback(error_callback)

        # Ensure data is at the broker before continuing
        producer.flush()
        print(f"Flushed {topic_name}")

    for chunk_idx, chunk_df in enumerate(ds_source):
        print(f"Chunk index: {chunk_idx}")
        print(chunk_df.head())
        print(chunk_df.describe())

        train_df, test_df = train_test_split(chunk_df, train_size=p_args["train_test_ratio"], shuffle=False)

        x_train_df = train_df.drop(["class"], axis=1)
        y_train_df = train_df["class"]

        x_test_df = test_df.drop(["class"], axis=1)
        y_test_df = test_df["class"]

        # The labels are set as the kafka message keys to store data
        # in multiple-partitions. Thus, enabling efficient data retrieval
        # using the consumer groups.
        x_train = list(filter(None, x_train_df.to_csv(index=False).split("\n")[1:]))
        y_train = list(filter(None, y_train_df.to_csv(index=False).split("\n")[1:]))

        x_test = list(filter(None, x_test_df.to_csv(index=False).split("\n")[1:]))
        y_test = list(filter(None, y_test_df.to_csv(index=False).split("\n")[1:]))

        write_to_kafka(p_args["train_topic"], zip(x_train, y_train))
        write_to_kafka(p_args["test_topic"], zip(x_test, y_test))

        recs_so_far: int = p_args["chunk_size"] * (chunk_idx + 1)
        print(f"Loaded {recs_so_far} records so far.")

        if recs_so_far >= max_records_to_load:
            print("Done")
            break
