============
Contributing
============

Contributions are welcome, and they are greatly appreciated! Every little bit
helps, and credit will always be given.

You can contribute in many ways:

Types of Contributions
----------------------

Report Bugs
~~~~~~~~~~~

Report bugs at https://github.com/maddataanalyst/hextractor/issues

If you are reporting a bug, please include:

* Your operating system name and version.
* Any details about your local setup that might be helpful in troubleshooting.
* Detailed steps to reproduce the bug.

Fix Bugs
~~~~~~~~

Look through the GitHub issues for bugs. Anything tagged with "bug"
and "help wanted" is open to whoever wants to implement a fix for it.

Implement Features
~~~~~~~~~~~~~~~~~~

Look through the GitHub issues for features. Anything tagged with "enhancement"
and "help wanted" is open to whoever wants to implement it.

Write Documentation
~~~~~~~~~~~~~~~~~~~

HeXtractor could always use more documentation, whether as part of
the official docs, in docstrings, or even on the web in blog posts, articles,
and such.

Submit Feedback
~~~~~~~~~~~~~~~

The best way to send feedback is to file an issue at
https://github.com/maddataanalyst/hextractor/issues.

If you are proposing a new feature:

* Explain in detail how it would work.
* Keep the scope as narrow as possible, to make it easier to implement.
* Remember that this is a volunteer-driven project, and that contributions
  are welcome :)

Get Started!
------------

Ready to contribute? Here's how to set up `hextractor` for local
development. Please note this documentation assumes you already have
`Git` installed and ready to go.

1. Fork the `hextractor` repo on GitHub.

::

2. Clone your fork locally:

   .. code-block:: bash

    $ cd path_for_the_repo
    $ git clone git@github.com:YOUR_NAME/hextractor.git

::

3. Set up your development environment. You can choose between Conda or a standard Python virtual environment:

   **Option 1: Using Conda**

   .. code-block:: bash

        # Create a new conda environment from the provided file
        $ conda env create -f environment.yml
        
        # Activate the environment
        $ conda activate hextractor
        
        # Install poetry inside the conda environment
        $ pip install poetry
        
        # Install the package with all dependencies
        $ poetry install --with dev --with research

   **Option 2: Using Standard Python Virtual Environment**

   .. code-block:: bash

        # Using venv (Python 3.3+)
        $ python -m venv hextractor-env
        $ source hextractor-env/bin/activate  # On Windows: hextractor-env\Scripts\activate
        
        # Or using virtualenv
        $ virtualenv hextractor-env
        $ source hextractor-env/bin/activate  # On Windows: hextractor-env\Scripts\activate
        
        # Install poetry
        $ pip install poetry
        
        # Install the package with all dependencies
        $ poetry install --with dev --with research

   This should change the shell to look something like:

   .. code-block:: bash

        (hextractor) $  # or (hextractor-env) $

::

4. Create a branch for local development:

   .. code-block:: bash

        $ git checkout -b name-of-your-bugfix-or-feature

   Now you can make your changes locally.

::

5. When you're done making changes, check that your changes pass the tests:

   .. code-block:: bash

        $ pytest ./tests

::

6. The next step would be to run all test cases. Before you run pytest, ensure all dependencies are installed:

   .. code-block:: bash

        # Dependencies should already be installed if you used poetry install as instructed above
        $ pytest ./tests

   If you get any errors while installing packages, try updating pip:

   .. code-block:: bash

        # Update pip
        $ pip install -U pip

::

7. Before raising a pull request you should also run tox. This will run the
   tests across different versions of Python:

   .. code-block:: bash

        $ tox

   If you are missing flake8, pytest and/or tox, just `pip install` them into
   your virtualenv.

::

8. If your contribution is a bug fix or new feature, you may want to add a test
   to the existing test suite. See section Add a New Test below for details.

::

9. Commit your changes and push your branch to GitHub:

   .. code-block:: bash

        $ git add .
        $ git commit -m "Your detailed description of your changes."
        $ git push origin name-of-your-bugfix-or-feature

::

10. Submit a pull request through the GitHub website.

::

Pull Request Guidelines
-----------------------

Before you submit a pull request, check that it meets these guidelines:

1. The pull request should include tests.

2. If the pull request adds functionality, the docs should be updated. Put your
   new functionality into a function with a docstring, and add the feature to
   the list in README.rst.

3. The pull request should work for Python 3.11 and for PyPy.

Add a New Test
--------------

When fixing a bug or adding features, it's good practice to add a test to
demonstrate your fix or new feature behaves as expected. These tests should
focus on one tiny bit of functionality and prove changes are correct.

To write and run your new test, follow these steps:

1. Add the new test to `tests/test_bake_project.py`. Focus your test on the
   specific bug or a small part of the new feature.

::

2. If you have already made changes to the code, stash your changes and confirm
   all your changes were stashed:

   .. code-block:: bash

        $ git stash
        $ git stash list

::

3. Run your test and confirm that your test fails. If your test does not fail,
   rewrite the test until it fails on the original code:

   .. code-block:: bash

        $ pytest ./tests

::

4. Proceed work on your bug fix or new feature or restore your changes. To
   restore your stashed changes and confirm their restoration:

   .. code-block:: bash

        $ git stash pop
        $ git stash list

::

5. Rerun your test and confirm that your test passes. If it passes,
   congratulations!

.. virtualenv: https://virtualenv.pypa.io/en/stable/installation
.. git: https://git-scm.com/book/en/v2/Getting-Started-Installing-Git
.. poetry: https://python-poetry.org/docs/#installation
