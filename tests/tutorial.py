## Necessary packages

import warnings

# 1. TimeGAN model
from timegan import timegan

# 2. Data loading
from timegan.data_loading import real_data_loading, sine_data_generation

# 3. Metrics
warnings.filterwarnings("ignore")

## Data loading
data_name = "stock"
seq_len = 24

if data_name in ["stock", "energy"]:
    ori_data = real_data_loading(data_name, seq_len)
elif data_name == "sine":
    # Set number of samples and its dimensions
    no, dim = 100, 5
    ori_data = sine_data_generation(no, seq_len, dim)

print(data_name + " dataset is ready.")
## Newtork parameters
parameters = dict()

parameters["module"] = "gru"
parameters["hidden_dim"] = 24
parameters["num_layer"] = 3
parameters["iterations"] = 100
parameters["batch_size"] = 128


# Run TimeGAN
generated_data = timegan.train_timegan(ori_data, parameters)
# timed_training = timegan.train_timegan_timed(ori_data, parameters, filename="timegan_save", seconds=60, phase=1, current_iter=0)


print("Finish Synthetic Data Generation")
