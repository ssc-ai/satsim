.. highlight:: shell

============
Contributing
============

Getting Started for Developers
------------------------------

Here's how to set up `satsim` for local development.

1. Clone the `satsim` repo on GitHub::

    $ git clone https://github.com/ssc-ai/satsim.git

2. Set up your local environment. Here are the install instructions for the
   following environments:

   virtualenvwrapper::

    $ mkvirtualenv satsim
    $ cd satsim/
    $ python -m pip install -U -r requirements_dev.txt
    $ python setup.py develop

   conda::

    $ conda create -n satsim
    $ conda activate satsim
    $ conda install tensorflow-gpu
    $ cd satsim/
    $ make develop

   docker::

    $ cd satsim/
    $ docker run -it -v $(pwd):/workspace/satsim algorithmhub/ahws-ipython3 bash
    $ cd /workspace/satsim
    $ make develop

   bare metal::

    $ cd satsim/
    $ make develop

3. Create a branch for local development::

    $ git checkout -b name-of-your-bugfix-or-feature

4. When you're done making changes, check that your changes pass flake8 and the
   tests, including testing other Python versions with tox::

    $ flake8 satsim tests
    $ python setup.py test -or- py.test -or- make test
    $ tox

5. Commit your changes and push your branch to GitHub::

    $ git add .
    $ git commit -m "Your detailed description of your changes."
    $ git push origin name-of-your-bugfix-or-feature

6. Submit a pull request at https://github.com/ssc-ai/satsim/pulls

Merge Request Guidelines
------------------------

Before you submit a merge request, check that it meets these guidelines:

1. The merge request should include tests.
2. If the merge request adds functionality, the docs should be updated. Put
   your new functionality into a function with a docstring.
3. The merge request should work for Python 3.5 and 3.6.

Tips
----

To run a subset of tests::

$ py.test tests.test_satsim

If your environment cannot find the `satsim` CLI, try relogging into the your
shell. Alternatively, try this if on a Linux system::

$ export PATH=$PATH:$HOME/.local/bin

Deploying
---------

A reminder for the maintainers on how to deploy.
Make sure all your changes are committed (including an entry in HISTORY.md).
Then run::

$ bumpversion patch # possible: major / minor / patch
$ git push
$ git push --tags

Other Ways of Contributing
--------------------------

Report Bugs
~~~~~~~~~~~

Report bugs at https://github.com/ssc-ai/satsim/issues.

If you are reporting a bug, please include:

* Your operating system name and version.
* Any details about your local setup that might be helpful in troubleshooting.
* Detailed steps to reproduce the bug.

Write Documentation
~~~~~~~~~~~~~~~~~~~

SatSim could always use more documentation, whether as part of the
official SatSim docs, in docstrings, or even on the web in blog posts,
articles, and such. To make the html documentation, run::

    $ make docs

HTML output will be in the directory, `satsim/docs/_build/html`.

Submit Feedback
~~~~~~~~~~~~~~~

The best way to send feedback is to file an issue at https://github.com/ssc-ai/satsim/issues.

If you are proposing a feature:

* Explain in detail how it would work.
* Keep the scope as narrow as possible, to make it easier to implement.
