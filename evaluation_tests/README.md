# Stream Clustering Robust to Concept Drift (Evaluation Experiments)

## Docker

Experiments can also be replicated by using Docker:

        sudo docker run --rm -v $(pwd)/results:/usr/src/app/results fiv5/sdostreamclust

Note that the previous command creates a [results] folder in your local machine in order to store the analysis results obtained from the container. 

*Warning!* the analysis process can take several days due to the high number of datasets, algorithms, and parameter adjustment processes. 


## Dependencies

These experiments have been run with **Python v3.8.14**.

To avoid conflicts with package-versions, we recommend using a clean python virtual environment:

        python3 -m venv venv

        source venv/bin/activate


Later, install dependencies by running:

        bash dependencies.sh

Finally, install SDOclustream as indicated in this repository, i.e.,

        pip3 install git+https://github.com/CN-TU/pysdoclust-stream.git@main

or download the main branch and run:

        pip3 install pysdoclust-stream-main.zip  
    
## Replication

To replicate evaluation experiments, simply run:

        bash run.sh 2> log.txt

It will generate two folders (if they don't exist already): [results] and [plots]

*Warning!* the analysis process can take several days due to the high number of datasets, algorithms, and parameter adjustment processes. 

## Results

The [results] folder contains results used in the paper, separated in three files:

- results_ex.csv (Multiple Concept Drift experiments)
- results_real.csv (Real Data experiments)
- results_syn.csv (Specific Concept Drift experiments)
