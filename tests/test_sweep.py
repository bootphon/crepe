import os
from pathlib import Path

import numpy as np
import crepe
from scipy.io import wavfile

# this data contains a sine sweep
file = Path(__file__).parent / 'sweep.wav'
f0_file = Path(__file__).parent / 'sweep.f0.csv'


def verify_f0():
    result = np.loadtxt(f0_file, delimiter=',', skiprows=1)

    # it should be confident enough about the presence of pitch in every frame
    assert np.mean(result[:, 2] > 0.5) > 0.98

    # the frequencies should be linear
    assert np.corrcoef(result[:, 1]) > 0.99

    os.remove(f0_file)


def test_sweep():
    crepe.process_file(str(file))
    verify_f0()

def test_sweep_predict():
    sr, data = wavfile.read(file)
    t, f0, confidence, activation = crepe.predict(data, sr, model_capacity="tiny")
    print("done")


def test_sweep_cli():
    assert os.system("crepe {}".format(file)) == 0
    verify_f0()
