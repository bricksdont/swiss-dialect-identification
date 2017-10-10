# swiss-dialect-identification
Baseline implementation for swiss dialect identification on the VarDial 2017 data set. Please use Python 2.X to run the script.

## Installation and Requirements

The script requires a recent version of the `scikit-learn` package. In most cases, installation is as easy as

    pip install scikit-learn

But see http://scikit-learn.org/stable/install.html for more detailed instructions.

Then clone the repository to your local computer or one of our servers:

    git clone https://github.com/bricksdont/swiss-dialect-identification

## Usage

To train a baseline model, use

    python baseline.py --train --model model_dummy.pkz --data train.csv --verbose --classifier dummy

To use a trained model to make predictions for the test samples:

    python baseline.py --predict --samples test.csv --model model_dummy.pkz > sandboxSubmission_dummy.csv

For other options, use `--help`:

    python baseline.py --help
