from __future__ import print_function
import sys, os
import tempfile
import shutil
sys.path.insert(1, os.path.join("..","..",".."))
import h2o
from tests import pyunit_utils
from h2o.automl import H2OAutoML
from h2o.automl.autoh2o import get_automl

def automl_checkpoints():

    df = h2o.import_file(path=pyunit_utils.locate("smalldata/logreg/prostate.csv"))

    # Split frames
    fr = df.split_frame(ratios=[.8,.1])

    # Set up train, validation, and test sets
    train = fr[0]
    valid = fr[1]
    test = fr[2]

    train["CAPSULE"] = train["CAPSULE"].asfactor()
    valid["CAPSULE"] = valid["CAPSULE"].asfactor()
    test["CAPSULE"] = test["CAPSULE"].asfactor()

    checkpoints_dir = tempfile.mkdtemp()

    aml = H2OAutoML(project_name="py_aml0", stopping_rounds=3, stopping_tolerance=0.001, stopping_metric="AUC",
                    max_models=2, seed=1234, export_checkpoints_dir=checkpoints_dir)
    aml.train(y="CAPSULE", training_frame=train)

    get_aml = get_automl(aml.project_name)
    num_files = len([f for f in os.listdir(checkpoints_dir) if "_cv" not in f])  # do not count CV models
    shutil.rmtree(checkpoints_dir)

    assert aml.project_name == get_aml["project_name"]
    assert num_files > 0, "No models generated by AutoML"
    assert get_aml["leaderboard"].nrows == num_files, "Not all generated autoML models were saved."


if __name__ == "__main__":
    pyunit_utils.standalone_test(automl_checkpoints)
else:
    automl_checkpoints()
