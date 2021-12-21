.. _csv_files:

Project Maintenance
===================

This page contains documentation for researchers who are maintaining their own repository of the project.

Releasing on PyPI
-----------------

The repository contains a workflow that automatically uploads a published GitHub release to PyPI.

Configuration
~~~~~~~~~~~~~

.. note:: You can skip this step if your repository is already configured and you want to go straight to the process of publishing a release.

1. Make necessary changes to `setup.py <../setup.py>`_. The fields :code:`author`, :code:`author_email`, and :code:`url` should be the most interesting. You'll probably also need to change the :code:`name` field to make uploading to PyPI possible.
2. On GitHub, go to the `action secrets <../../../settings/secrets/actions>`_ settings and add your PyPI `API token <https://pypi.org/help/#apitoken>`_ as :code:`PYPI_PASSWORD`.

Publishing a Release
~~~~~~~~~~~~~~~~~~~~

1. In  `setup.py <../setup.py>`_, change the field :code:`version` to the version number of your release. Look at `this page <https://the-hitchhikers-guide-to-packaging.readthedocs.io/en/latest/specification.html>`_ if you're unsure.
2. Make sure `actions are enabled <../../../settings/actions>`_.
3. Create a new release on GitHub, preferably using a tag with the same version number from before, and publish it. You can check the PyPI release workflow `here <../../../actions/workflows/publish-package.yml>`_. If a green tick shows up, the PyPI release should be up!
