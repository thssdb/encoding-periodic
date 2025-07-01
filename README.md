# FLEA: Frequency-based Lossless Encoding Algorithm for Periodic Time Series

This repository contains the experimental implementation and evaluation code for the paper "FLEA: Frequency-based Lossless Encoding Algorithm for Periodic Time Series."

## Abstract

Existing lossless time series compressors surprisingly ignore the strong periodic pattern,
a common characteristic that limits their effectiveness on real-world data.
To address this, we design FLEA (Frequency-based Lossless Encoding Algorithm),
a novel lossless encoding scheme that exploits the frequency domain for periodic time series.
While directly storing the extremely high precision frequency coefficients is obviously not an option to lossless compression, 
we propose to quantize the frequency coefficients and capture the corresponding low precision residuals in time domain.
The core challenge is thus to optimize the trade-off between the encoding costs of the quantized frequency component and its time-domain residual.
We address this by modeling it as a rate-optimal search problem, made tractable by an energy-based cost model that guides the global search for the optimal parameter.
Moreover,
FLEA introduces two adaptive encoders that
(1) horizontally partition the frequency component for skewed and sparse coefficients,
and 
(2) vertically partition the residual bit-width for the long-tailed distribution.
Extensive experiments on a diverse suite of real-world datasets show that FLEA establishes a new state-of-the-art in compression ratio,
particularly on its target periodic data, with an average improvement of 10.2% over the runner-up.
Its highly efficient decoding, approximately twice as fast as its encoding, 
makes FLEA practical for write-once-read-many scenarios of databases, 
leading to the native implementation in Apache IoTDB.

## System Requirements

- **Python**: 3.10 or higher (tested on Python 3.12)
- **Package Manager**: Poetry
- **Operating System**: Cross-platform (Linux, macOS, Windows)

## Dataset Organization

The repository includes public datasets stored in Git LFS format for reproducibility:

- **`./data/`**: Periodic time series datasets
- **`./data_no_period/`**: Aperiodic time series datasets

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/thssdb/encoding-periodic.git
   ```

2. **Install dependencies**:
   ```bash
   poetry install
   ```

3. **Activate the virtual environment** (if needed):
   ```bash
   poetry shell
   ```

## Usage

### Running Experiments

Execute the experimental scripts from the repository root directory:

```bash
python <experiment_script>.py
```

### Output Structure

- **Experimental Results**: Generated in `./exp_results/`
- **Visualizations**: Generated in `./figures/`

## License

This project is licensed under GPL-3.0-or-later - see the LICENSE file for details.