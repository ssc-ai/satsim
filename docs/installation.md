```{highlight} shell
```

# Installation

## Stable Release

To install SatSim:

```console
$ pip3 install satsim
```

Or build python wheel file from [GitHub releases]:

Then run this command in your terminal from the location of the python wheel
file.

```console
$ pip3 install satsim-VERSION-py2.py3-none-any.whl
```

If you don't have [pip] installed, this [Python installation guide] can guide
you through the process.

## From Sources

The sources for SatSim can be downloaded from the [GitHub repo].

You can either clone the repository:

```console
$ git -c clone https://github.com/ssc-ai/satsim.git
```

Or download the [tarball]:

```console
$ curl -k -OL https://github.com/ssc-ai/satsim/archive/refs/heads/master.zip
```

Once you have a copy of the source, change directory into SatSim and install
it with:

```console
$ python3 setup.py install -or- make install (may require sudo or --user)
```

If you will be modifying code (see contributing page for more details), you
should install with:

```console
$ python3 setup.py develop -or- make develop
```

[github releases]: https://github.com/ssc-ai/satsim/releases
[github repo]: https://github.com/ssc-ai/satsim.git
[pip]: https://pip.pypa.io
[python installation guide]: http://docs.python-guide.org/en/latest/starting/installation/
[tarball]: https://github.com/ssc-ai/satsim/archive/refs/heads/master.zip
