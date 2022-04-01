.. highlight:: shell

============
Installation
============


Stable Release
--------------

To install SatSim, download the python wheel file from `GitLab releases`_:

.. _GitLab releases: https://gitlab.pacificds.com/machine-learning/satsim/releases


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

The sources for SatSim can be downloaded from the `GitLab repo`_.

You can either clone the repository:

.. code-block:: console

    $ git -c http.sslVerify=false clone https://gitlab.pacificds.com/machine-learning/satsim.git

Or download the `tarball`_:

.. code-block:: console

    $ curl -k -OL https://gitlab.pacificds.com/machine-learning/satsim/-/archive/master/satsim-master.tar.gz

Once you have a copy of the source, change directory into SatSim and install
it with:

.. code-block:: console

    $ python3 setup.py install -or- make install (may require sudo or --user)

If you will be modifying code (see contributing page for more details), you
should install with:

.. code-block:: console

    $ python3 setup.py develop -or- make develop

.. _GitLab repo: https://gitlab.pacificds.com/machine-learning/satsim.git
.. _tarball: https://gitlab.pacificds.com/machine-learning/satsim/-/archive/master/satsim-master.tar.gz
