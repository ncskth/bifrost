# Bifrost

**Bifrost** translates models in [PyTorch](https://pytorch.org) into [SpiNNaker](https://spinnakermanchester.github.io/) executables.

It is named after Bifrost, the bridge between middle earth and the realms of the gods in Norse mythology.

## Usage: PyTorch -> PyNN script

After training the model, when saving the `StateDictionary` you need to use the utility `set_parameter_buffers` included in `bifrost.extract.torch.parameter_buffers`.
To load the model again, the `torch.nn.Module` method `load_state_dict` has to have the argument `strict` set to `False`. 

Models are encoded by providing the Python import path to the model class as well as the shape of input tensor (here with 1 timestep, 8 batches, 2 channels, and a 640x480 image input):

```bash
bifrost model.SNNModel "(1, 8, 2, 640, 480)" > output.py
```

## Usage: Execute PyNN script with weights file 

The output can now be evaluated with a specific set of parameter values from a [PyTorch Lightning](https://pytorch-lightning.readthedocs.io/en/latest/) checkpoint file. Note that the command must run inside a SpiNNaker-friendly environment (see below):

```bash
python output.py weights.ckpt 
```

By "SpiNNaker-friendly environment" we mean a working installation of [sPyNNaker](https://github.com/SpiNNakerManchester/sPyNNaker) and access to a SpiNNaker machine.
See the attached `Dockerfile` for a quick environment installation.

## Credits

Bifrost is maintained by 

* [Jens E. Pedersen](https://www.kth.se/profile/jeped) (@GitHub [jegp](https://github.com/jegp/)), doctoral student at KTH Royal Institute of Technology, Sweden.
* [Garibaldi Pineda Garcia](https://profiles.sussex.ac.uk/p467078-garibaldi-pineda-garcia) (@GitHub [chanokin](https://github.com/chanokin/)), PostDoc at The University of Sussex, UK.

The project is indebted to the work by [Petrut A. Bogdan](https://github.com/pabogdan/) on JSON conversions from ANN to SNN and to [Simon Davidson](http://apt.cs.manchester.ac.uk/people/sdavidson/) for advice and supervision.

The work has received funding from the EC Horizon 2020 Framework Programme under Grant Agreements 785907 and 945539 (HBP)

## License
LGPLv3. See [LICENSE](LICENSE) for license details.