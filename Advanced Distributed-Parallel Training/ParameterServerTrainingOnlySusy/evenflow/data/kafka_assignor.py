import collections
import itertools
from typing import Optional

import kafka
import six
from kafka import TopicPartition
from kafka.coordinator.assignors.abstract import AbstractPartitionAssignor
from kafka.coordinator.protocol import (
    ConsumerProtocolMemberAssignment,
    ConsumerProtocolMemberMetadata,
)


class CustomPartitionAssignor(AbstractPartitionAssignor):
    DEFAULT_GENERATION_ID: int = -1

    name = "custom1"
    version = 0

    member_assignment = None
    generation = DEFAULT_GENERATION_ID

    _latest_partition_movements = None

    def __init__(self, world_size: int, rank: int) -> None:
        """

        Args:
            world_size:
            rank:
        """
        super().__init__()
        self.rank: int = rank
        self.world_size: int = world_size
        self.version: int = 0

    def assign(self, cluster: kafka.cluster.ClusterMetadata, member_metadata: dict[str, any]) -> dict[str, any]:
        """Performs group assignment given cluster metadata and member subscriptions

        Arguments:
            cluster (ClusterMetadata): cluster metadata
            member_metadata (dict of {member_id: MemberMetadata}): decoded metadata for each member in the group.

        Returns:
          dict: {member_id: MemberAssignment}
        """
        all_topics: set[str] = set()
        for metadata in six.itervalues(member_metadata):
            all_topics.update(metadata.subscription)

        all_topic_partitions: list[TopicPartition] = []
        for topic in all_topics:
            partitions: Optional[set[int]] = cluster.partitions_for_topic(topic)

            if partitions is None:
                raise RuntimeError(f"No partition metadata for topic {topic}")

            for partition in partitions:
                # Workers are assigned equal chunks of partitions.
                # PS is assigned all partitions
                if partition % (self.world_size - 1) == self.rank - 1 or self.rank == 0:
                    all_topic_partitions.append(TopicPartition(topic, partition))

        all_topic_partitions.sort()

        # construct {member_id: {topic: [partition, ...]}}
        assignment: dict[str, dict] = collections.defaultdict(lambda: collections.defaultdict(list))

        member_iter = itertools.cycle(sorted(member_metadata.keys()))
        for partition in all_topic_partitions:
            member_id = next(member_iter)

            # Because we constructed all_topic_partitions from the set of
            # member subscribed topics, we should be safe assuming that
            # each topic in all_topic_partitions is in at least one member
            # subscription; otherwise this could yield an infinite loop
            while partition.topic not in member_metadata[member_id].subscription:
                member_id = next(member_iter)

            assignment[member_id][partition.topic].append(partition.partition)

        protocol_assignment: dict[str, any] = {}
        for member_id in member_metadata:
            protocol_assignment[member_id] = ConsumerProtocolMemberAssignment(
                self.version, sorted(assignment[member_id].items()), b""
            )

        return protocol_assignment

    def parse_member_metadata(self, metadata: any) -> None:
        """
        Parses member metadata into a python object.
        This implementation only serializes and deserializes the StickyAssignorMemberMetadataV1 user data,
        since no StickyAssignor written in Python was deployed ever in the wild with version V0, meaning that
        there is no need to support backward compatibility with V0.

        Arguments:
          metadata (MemberMetadata): decoded metadata for a member of the group.

        Returns:
          parsed metadata (StickyAssignorMemberMetadataV1)
        """
        pass

    def metadata(self, topics):
        return ConsumerProtocolMemberMetadata(self.version, list(topics), b"")

    def on_assignment(self, assignment):
        if not assignment.assignment[0]:
            raise RuntimeError("No topics available.")
        topic: str = assignment.assignment[0][0]
        print(
            f"Assigned partitions to rank {self.rank}: {[x[1] for x in assignment.partitions()]} for topic [{topic}]."
        )

    def on_generation_assignment(self, generation):
        pass
