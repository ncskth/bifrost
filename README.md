# Bifrost

**Bifrost** translates models in [PyTorch](https://pytorch.org) into [SpiNNaker](https://spinnakermanchester.github.io/) executables.

It is named after Bifrost, the bridge between middle earth and the realms of the gods in Norse mythology.

## Usage

Models are encoded by providing the Python import path to the model class as well as parameter values from a [PyTorch Lightning](https://pytorch-lightning.readthedocs.io/en/latest/) checkpoints file:

```bash
python bifrost.py model.SNNModel > output.py
```

The output can now be evaluated inside a SpiNNaker-friendly environment (see below):

```bash
python output.py
```

By "SpiNNaker-friendly environment" we mean need a working installation of [sPyNNaker](https://github.com/SpiNNakerManchester/sPyNNaker) and access to a SpiNNaker machine.
See the attached `Dockerfile` for a quick environment installation.

## Credits

Bifrost is maintained by 

* [Jens E. Pedersen](https://www.kth.se/profile/jeped) (@GitHub [jegp](https://github.com/jegp/)), doctoral student at KTH Royal Institute of Technology, Sweden.

The project is indebted to the work by [Petrut A. Bogdan](https://github.com/pabogdan/) on JSON conversions from ANN to SNN.

The work has received funding from the EC Horizon 2020 Framework Programme under Grant Agreements 785907 and 945539 (HBP)

## License
LGPLv3. See [LICENSE](LICENSE) for license details.