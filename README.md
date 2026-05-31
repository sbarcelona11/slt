# Sign Language Transformers (CVPR'20)

This repo contains the training and evaluation code for the paper [Sign Language Transformers: Sign Language Transformers: Joint End-to-end Sign Language Recognition and Translation](https://www.cihancamgoz.com/pub/camgoz2020cvpr.pdf). 

This code is based on [Joey NMT](https://github.com/joeynmt/joeynmt) but modified to realize joint continuous sign language recognition and translation. For text-to-text translation experiments, you can use the original Joey NMT framework.
 
## Requirements
* Download the feature files using the `data/download.sh` script.

* [Optional] Create a conda or python virtual environment.

* Install required packages using the `requirements.txt` file.

    `pip install -r requirements.txt`

## Usage

  `python -m signjoey train configs/sign.yaml`

### Device selection (CPU/CUDA/MPS)

- Config: set `training.device` to `cpu`, `cuda`, `mps`, or `auto` (default behavior is backwards compatible with `training.use_cuda`).
- CLI override: `python -m signjoey train configs/sign.yaml --device mps`
- CUDA GPU selection: `python -m signjoey train configs/sign.yaml --device cuda --gpu_id 0`

Note: the pinned `requirements.txt` uses `torch==1.4.0`, which does **not** support MPS. To run on Apple Silicon MPS you need a newer PyTorch build (and likely a compatible `torchtext`), otherwise `--device mps` / `training.device: mps` will fail at runtime.

### macOS (Apple Silicon) + MPS setup

The original codebase relies on torchtext's *legacy* `Field`/`BucketIterator` APIs. Those APIs were removed from modern torchtext releases, and torchtext has strict version coupling to torch (see the PyTorch domain compatibility matrix).

To make MPS feasible without torchtext, this repo includes a torchtext-free data loader:

- Set `data.loader: native` in your YAML (defaults to `torchtext` if omitted).
- Install a modern PyTorch and the minimal deps: `pip install -r requirements-macos-mps.txt`

Example run:

`python -m signjoey train configs/sign.yaml --device mps`

and ensure your YAML has:

`training.device: mps`
`data.loader: native`

! Note that the default data directory is `./data`. If you download them to somewhere else, you need to update the `data_path` parameters in your config file.   
## ToDo:

- [X] *Initial code release.*
- [X] *Release image features for Phoenix2014T.*
- [ ] Share extensive qualitative and quantitative results & config files to generate them.
- [ ] (Nice to have) - Guide to set up conda environment and docker image.

## Reference

Please cite the paper below if you use this code in your research:

    @inproceedings{camgoz2020sign,
      author = {Necati Cihan Camgoz and Oscar Koller and Simon Hadfield and Richard Bowden},
      title = {Sign Language Transformers: Joint End-to-end Sign Language Recognition and Translation},
      booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
      year = {2020}
    }

## Acknowledgements
<sub>This work was funded by the SNSF Sinergia project "Scalable Multimodal Sign Language Technology for Sign Language Learning and Assessment" (SMILE) grant agreement number CRSII2 160811 and the European Union’s Horizon2020 research and innovation programme under grant agreement no. 762021 (Content4All). This work reflects only the author’s view and the Commission is not responsible for any use that may be made of the information it contains. We would also like to thank NVIDIA Corporation for their GPU grant. </sub>
