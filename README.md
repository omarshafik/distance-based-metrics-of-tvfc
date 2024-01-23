# Distance-Based Metrics of Time-Varying Functional Connectivity
Complete python script and tools of all study procedures of "Distance-Based Metrics of Time-Varying Functional Connectivity"

## Install
To install the dependencies of the project, you need to install pdm. Go to the [official website](https://pdm-project.org/latest) and follow the directions.
After installing pdm, you can install the dependencies by running `python -m pdm install`

## Run
To run the study procedures and create a results directory under the parent directory of the given input directory, run the command (assuming you are opening the terminal from the top repository directory):
`python .\src\run.py -s 1705960800 --indir /path/to/parcels/timeseries/directory`