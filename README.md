# swiss-dialect-identification
Baseline implementation for swiss dialect identification on the VarDial 2017 data set. Please use Python 2.X to run the script.

To train a baseline model, use

    python baseline.py --train --model model_dummy.pkz --data train.csv --verbose --classifier dummy

To use a trained model to make predictions for the test samples:

    python baseline.py --predict --samples test.csv --model model_dummy.pkz > sandboxSubmission_dummy.csv

For other options, use `--help`:

    python baseline.py --help
