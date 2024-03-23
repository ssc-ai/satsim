.. highlight:: shell

============
Installation
============


Stable Release
--------------

To install SatSim:

.. code-block:: console

    $ pip3 install satsim

Or build python wheel file from `GitHub releases`_:

.. _GitHub releases: https://github.com/ssc-ai/satsim/releases

Then run this command in your terminal from the location of the python wheel
file.

.. code-block:: console

    $ pip3 install satsim-VERSION-py2.py3-none-any.whl


If you don't have `pip`_ installed, this `Python installation guide`_ can guide
you through the process.

.. _pip: https://pip.pypa.io
.. _Python installation guide: http://docs.python-guide.org/en/latest/starting/installation/


From Sources
------------

The sources for SatSim can be downloaded from the `GitHub repo`_.

You can either clone the repository:

.. code-block:: console

    $ git -c clone https://github.com/ssc-ai/satsim.git

Or download the `tarball`_:

.. code-block:: console

    $ curl -k -OL https://github.com/ssc-ai/satsim/archive/refs/heads/master.zip

Once you have a copy of the source, change directory into SatSim and install
it with:

.. code-block:: console

    $ python3 setup.py install -or- make install (may require sudo or --user)

If you will be modifying code (see contributing page for more details), you
should install with:

.. code-block:: console

    $ python3 setup.py develop -or- make develop

.. _GitHub repo: https://github.com/ssc-ai/satsim.git
.. _tarball: https://github.com/ssc-ai/satsim/archive/refs/heads/master.zip
