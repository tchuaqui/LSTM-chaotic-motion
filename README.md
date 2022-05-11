# LSTM-chaotic-motion
LSTM model is used to forecast the chaotic motion of a double pendulum.

A sequence-to-vector LSTM model is trained on time series data of the chaotic motion of a double pendulum. The outputs of the model are the angular positions and velocities of the two masses of the pendulum (4-dimensional vector).
For training, a custom loss function, which penalises solutions that violate energy conservation principles, is used. 

The loss function combines a first term J1, corresponding to standard MSE of the model outputs, with a second penalty term J2, computed from the error in the system's internal energy (only kinetic + potential energy are considered). A hyperparameter alpha (between 0 and 1) is used to weigh the contribution of the penalty term, with 0 corresponding to plain MSE (J1 term) and 1 corresponding to the J2 penalty term only.

Training and validation data is obtained by solving the ordinary differential equations governing the pedulum's motion and stored in /data/ directory using pre-defined properties (masses, lengths and gravitational acceleration). Only 50 examples are included in /data/ but LSTM was originally trained/validated using data from 2500 files. More and/or alternative data using a different set of physical properties can be easily generated from the main.py script. The original 2500 files can be retrieved by deleting the current 50 files in /data/ and running main.py with generate_new_data=True, without changing any of the remaining default parameters. Other problem inputs (including hyperparameters for the LSTM model) can also be defined in main.py. Trained models can be re-used setting use_trained=True in main.py, or alternatively, re-trained from different data/parameters using use_trained=False. Note that re-training will overwrite any existing saved models.

The accuracy of the LSTM forecast is determined from the MSE of the angular positions of the two masses (this differs from the J1 term above as it excludes the velocity terms). Increasing alpha does not necessarily result in better accuracy on the validation set, as shown below (MSE of angular positions over 100 epochs, with blue -> alpha=0; green -> alpha=0.1; magenta -> alpha=0.5):
<img src=https://user-images.githubusercontent.com/26413615/167838344-3d7fdc97-fced-46db-b9da-e56849369720.svg width="400" height="400">

Despite this, using alpha=0 results in very erratic predictions, clearly violating energy conservation laws, as shown in the following examples taken from the validation set:

alpha=0

https://user-images.githubusercontent.com/26413615/167836411-2db59fb1-4287-4843-9146-f64797a95863.mov

alpha = 0.1 (looks better - smoother response)

https://user-images.githubusercontent.com/26413615/167836456-e59d97f6-1c90-4dc4-8429-934d87a9ad36.mov

alpha = 0.5 (chaotic motion is lost)

https://user-images.githubusercontent.com/26413615/167836478-80b09fa0-a2e1-42ea-b180-ace242439863.mov

Increasing alpha by a significant amount results in a smoother response, ensuring energy conservation, but at the expense of losing predictive accuracy (in the case above with alpha=0.5 the chaotic motion was lost completely). A small value of alpha is therefore recommended.

