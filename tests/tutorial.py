"""Fast demo run -- trains a real (small) TimeGAN model on the bundled
stock dataset. Not a pytest test (see tests/test_training.py for those);
run directly with `python3 tests/tutorial.py`.

Guarded behind `if __name__ == "__main__"` so importing this module (e.g.
pytest's `--doctest-modules` collection, which imports every .py under
`tests/` to scan for doctests) doesn't accidentally kick off a real
training run as a side effect -- confirmed this was happening before the
guard was added: `pytest --doctest-modules tests` hung importing this
file, since its logic used to run unconditionally at module level.
"""

import warnings

from timegan import timegan
from timegan.data_loading import real_data_loading, sine_data_generation


def main():
    warnings.filterwarnings("ignore")

    data_name = "stock"
    seq_len = 24

    if data_name in ["stock", "energy"]:
        ori_data = real_data_loading(data_name, seq_len)
    elif data_name == "sine":
        no, dim = 100, 5
        ori_data = sine_data_generation(no, seq_len, dim)

    print(data_name + " dataset is ready.")

    parameters = dict()
    parameters["module"] = "gru"
    parameters["hidden_dim"] = 24
    parameters["num_layer"] = 3
    parameters["iterations"] = 100
    parameters["batch_size"] = 128

    timegan.train_timegan(ori_data, parameters)
    print("Finish Synthetic Data Generation")


if __name__ == "__main__":
    main()
