# Contributing to TRAIT2D

Everyone with interest in the project is welcome to contribute and we are thankful for all contributions that are made to the tool.

The following guidelines will explain the contribution process.

## Table of Contents

* [Who Can Contribute](#who-can-contribute)
* [Reporting Issues](#reporting-issues)
* [Contributing Code](#contributing-code)
* [Contributing Documentation](#contributing-documentation)
* [Requesting Features](#requesting-features)
* [Starting Discussions](#starting-discussions)

## Who Can Contribute

Contributions are open to everyone, regardless of background, with interest in the tool. Contributions are not strictly limited to code but can also be made in the form of reporting issues and bugs, adding to the documentation, requesting features you think could add to the tool or starting discussions.

## Reporting Issues

If you come across a problem when using the tool or suspect something isn't working as it should, it is much appreciated if you raise an issue on GitHub. You can create a new issue by either following [this link](https://github.com/Eggeling-Lab-Microscope-Software/TRAIT2D/issues/new/choose) or navigating to the [issues](https://github.com/Eggeling-Lab-Microscope-Software/TRAIT2D/issues) tab on the [main repository](https://github.com/Eggeling-Lab-Microscope-Software/TRAIT2D) and clicking `New issue`.

Before raising a new issue you should check that the issue hasn't already been submitted by someone else. You can do so by navigating to the [issues](https://github.com/Eggeling-Lab-Microscope-Software/TRAIT2D/issues) tab on the [main repository](https://github.com/Eggeling-Lab-Microscope-Software/TRAIT2D) and using the search bar to look for similar issues. If you are in doubt whether you're issue is unique, please go ahead an submit it anyways; things can always be sorted out afterwards.

A good issue should do the following:

* Specify the system (such as processor, operating system) you are running the tool on.
* Specify how to reproduce the issue.
* If applicable, contain the error message that is produced (you can simply copy-paste this).
* If applicable, contain example code (such as a notebook or a Python script) that produces the error.

If you have something to add to an already existing issue you can always do so by commenting on it. If you are experiencing an already existing issue but have nothing new to add, you can add a [reaction](https://github.blog/2016-03-10-add-reactions-to-pull-requests-issues-and-comments/) to it.

## Contributing Code

Contributing code can be done directly via pull requests. If your pull request fixes an issue that hasn't bee raised yet, please also [create a separate issue](#reporting-issues) so others can easily find it.

Pull requests can be made by directly following [this link](https://github.com/Eggeling-Lab-Microscope-Software/TRAIT2D/compare), provided you have [forked](https://docs.github.com/en/get-started/quickstart/fork-a-repo) the [main repository](https://github.com/Eggeling-Lab-Microscope-Software/TRAIT2D) and implemented the changes on a branch of the fork already. You can also navigate to the [pull request](https://github.com/Eggeling-Lab-Microscope-Software/TRAIT2D/pulls) tab or initiate the pull request from your forked repository.

A good pull request should:

* Specify why the changes are required -OR- reference the issue(s) it fixes
* Briefly explain the changes made, both in the pull request itself as well as in the commit messages.
* Not consist of too many fragmented commits.
* Not address too many unrelated issues. Consider making separate pull requests instead.

If you have an idea for a new code addition but are unsure on the details or how to implement it, you can also [start a discussion](#starting-discussions) to discuss your idea or create a [draft pull request](https://github.blog/2019-02-14-introducing-draft-pull-requests/).

## Contributing Documentation

If you see something in the [documentation](https://eggeling-lab-microscope-software.github.io/TRAIT2D/) that you think needs improvement or contains an error, simply click the `Edit on GitHub` button in the top right corner. This will allow you to edit the corresponding file directly in your browser and create a pull request with any changes you've made. As explained in the section about [contributing code](#contributing-code), please explain clearly why the change is necessary and choose an informative commit message.

In order to add new pages to the documentation, you will need to fork the repository and create a new file for the page manually. All documentation pages are located inside the [sphinx/source](https://github.com/Eggeling-Lab-Microscope-Software/TRAIT2D/tree/master/sphinx/source) directory as [reStructuredText](https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html) (`.rst`) files. Do not forget to reference your newly created file in [`index.rst`](https://github.com/Eggeling-Lab-Microscope-Software/TRAIT2D/blob/master/sphinx/source/index.rst). If you are unsure how to structure and reference your documentation page, [`tracker_gui.rst`](https://raw.githubusercontent.com/Eggeling-Lab-Microscope-Software/TRAIT2D/master/sphinx/source/tracker_gui.rst) is a simple page that you can use as a starting point.

## Requesting Features

If you have an idea for a feature that you think could add to the usefulness of the tool, you are welcome to create a feature request by submitting a [new issue](https://github.com/Eggeling-Lab-Microscope-Software/TRAIT2D/issues/new/choose].

A feature request should specify:

* *Why* you think the feature is necessary.
* *What* the feature is. Please be as detailed as you can when requesting a new feature.

If you already have some ideas on how to implement feature, please also specify:

* *How* the feature can be implemented (algorithms, code examples).
* If you'd like to work on the feature yourself.

## Starting Discussions

Even if you do not have an issue to report, you are welcome to start a discussion. There are two ways to start a discussion. You can either create a new discussion under the [discussion tab](https://github.com/Eggeling-Lab-Microscope-Software/TRAIT2D/discussions). However, since this is a relatively new feature, you are also welcome to just create a [new issue](https://github.com/Eggeling-Lab-Microscope-Software/TRAIT2D/issues/new/choose). A `discussion` label will then be added to the issue by a maintainer.

There are no special requirements for starting discussions except of course that they should be relevant to the tool.