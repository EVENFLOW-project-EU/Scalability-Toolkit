!/bin/bash

./kafka_2.13-3.1.2/bin/zookeeper-server-start.sh -daemon ./kafka_2.13-3.1.2/config/zookeeper.properties
./kafka_2.13-3.1.2/bin/kafka-server-start.sh -daemon ./kafka_2.13-3.1.2/config/server.properties
