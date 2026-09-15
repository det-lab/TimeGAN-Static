# mypy: ignore-errors
"""Full demo run -- trains a production-scale TimeGAN model on the bundled
stock dataset (10,000 iterations) and reports discriminative/predictive
scores plus PCA/t-SNE visualizations. Not a pytest test (see
tests/test_training.py for those); run directly with
`python3 tests/full_demo.py`. Slow by design -- this is meant to produce
a real result, not to run in CI.

Guarded behind `if __name__ == "__main__"` so importing this module (e.g.
pytest's `--doctest-modules` collection, which imports every .py under
`tests/` to scan for doctests) doesn't accidentally kick off this full
10,000-iteration run as a side effect -- confirmed this was happening
before the guard was added: `pytest --doctest-modules tests` hung
importing this file, since its logic used to run unconditionally at
module level.
"""

import warnings

import numpy as np

from timegan import timegan
from timegan.data_loading import real_data_loading, sine_data_generation
from timegan.metrics.discriminative_metrics import discriminative_score_metrics
from timegan.metrics.predictive_metrics import predictive_score_metrics
from timegan.metrics.visualization_metrics import visualization


def main():
    warnings.filterwarnings("ignore")

    data_name = "stock"
    seq_len = 24

    if data_name in ["stock", "energy"]:
        ori_data = real_data_loading(data_name, seq_len)
    elif data_name == "sine":
        no, dim = 10000, 5
        ori_data = sine_data_generation(no, seq_len, dim)

    print(data_name + " dataset is ready.")

    parameters = dict()
    parameters["module"] = "gru"
    parameters["hidden_dim"] = 24
    parameters["num_layer"] = 3
    parameters["iterations"] = 10000
    parameters["batch_size"] = 128

    generated_data = timegan.train_timegan(ori_data, parameters)
    print("Finish Synthetic Data Generation")

    metric_iteration = 5

    discriminative_score = list()
    for _ in range(metric_iteration):
        temp_disc = discriminative_score_metrics(ori_data, generated_data)
        discriminative_score.append(temp_disc)

    print("Discriminative score: " + str(np.round(np.mean(discriminative_score), 4)))

    predictive_score = list()
    for _ in range(metric_iteration):
        temp_pred = predictive_score_metrics(ori_data, generated_data)
        predictive_score.append(temp_pred)

    print("Predictive score: " + str(np.round(np.mean(predictive_score), 4)))

    visualization(ori_data, generated_data, "pca")
    visualization(ori_data, generated_data, "tsne")


if __name__ == "__main__":
    main()
