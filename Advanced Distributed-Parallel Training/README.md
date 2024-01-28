# Setup - repository

Clone this repository and stay on the master branch.
The main driver script can be found under the following path and requires a few steps in order to run.

Driver

    evenflow/driver.py


Steps to ensure that the project can be run.

1. Download the SUSY dataset (see nb_driver.py).
2. Setup a Kafka cluster.
3. Run the preprocessing script and ensure that Kafka topics have been created.

# Setup - notebook

Simply download the nb_driver.py notebook and place the following env variables in a .env file in the same directory.

    GIT_PAT=XXX


Requirements
* ipyparallel
* jdk-11/14/17
* pip

# PS tests

For a working PS example check the evenflow/pytorch/approach2 folder and it's README under evenflow/pytorch/approach2/resources.


https://mvnrepository.com/artifact/org.apache.kafka/kafka-clients/3.1.2
https://gist.github.com/mwufi/6718b30761cd109f9aff04c5144eb885

## Kafka

Setup

    docker-compose -f docker-compose.yml up kafka1 zookeeper


Read JSON messages from the topic (max 10).

    kafkacat -G test-group-1 -b localhost:9092 -t train -C -J -c 10 -o beginning | jq '.'



## Debugging with IDEA
        
```python
import pydevd_pycharm

pydevd_pycharm.settrace('localhost', port=1234, stdoutToServer=True, stderrToServer=True)
```
