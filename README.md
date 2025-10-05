# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/stevenmburns/antenna_designer/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                                |    Stmts |     Miss |   Cover |   Missing |
|---------------------------------------------------- | -------: | -------: | ------: | --------: |
| src/antenna\_designer/\_\_init\_\_.py               |        7 |        0 |    100% |           |
| src/antenna\_designer/\_\_main\_\_.py               |        0 |        0 |    100% |           |
| src/antenna\_designer/builder.py                    |      131 |       28 |     79% |150-151, 154-190 |
| src/antenna\_designer/cli.py                        |       96 |        0 |    100% |           |
| src/antenna\_designer/core.py                       |        6 |        0 |    100% |           |
| src/antenna\_designer/designs/bowtie.py             |       26 |        0 |    100% |           |
| src/antenna\_designer/designs/bowtiearray2x4.py     |        7 |        0 |    100% |           |
| src/antenna\_designer/designs/bowtiearray.py        |        7 |        0 |    100% |           |
| src/antenna\_designer/designs/dipole.py             |       17 |        0 |    100% |           |
| src/antenna\_designer/designs/fandipole.py          |       59 |       13 |     78% |    95-114 |
| src/antenna\_designer/designs/freq\_based/invvee.py |       14 |        0 |    100% |           |
| src/antenna\_designer/designs/freq\_based/yagi.py   |       38 |        0 |    100% |           |
| src/antenna\_designer/designs/hexbeam.py            |       45 |        0 |    100% |           |
| src/antenna\_designer/designs/invvee.py             |       13 |        0 |    100% |           |
| src/antenna\_designer/designs/invveearray.py        |        8 |        0 |    100% |           |
| src/antenna\_designer/designs/moxon.py              |       33 |        0 |    100% |           |
| src/antenna\_designer/designs/rawdipole.py          |       12 |        0 |    100% |           |
| src/antenna\_designer/designs/vertical.py           |       24 |        0 |    100% |           |
| src/antenna\_designer/far\_field.py                 |      105 |        0 |    100% |           |
| src/antenna\_designer/opt.py                        |       39 |        1 |     97% |        53 |
| src/antenna\_designer/pysim.py                      |      259 |        0 |    100% |           |
| src/antenna\_designer/sim.py                        |       42 |        0 |    100% |           |
| src/antenna\_designer/sweep.py                      |      117 |        0 |    100% |           |
|                                           **TOTAL** | **1105** |   **42** | **96%** |           |


## Setup coverage badge

Below are examples of the badges you can use in your main branch `README` file.

### Direct image

[![Coverage badge](https://raw.githubusercontent.com/stevenmburns/antenna_designer/python-coverage-comment-action-data/badge.svg)](https://htmlpreview.github.io/?https://github.com/stevenmburns/antenna_designer/blob/python-coverage-comment-action-data/htmlcov/index.html)

This is the one to use if your repository is private or if you don't want to customize anything.

### [Shields.io](https://shields.io) Json Endpoint

[![Coverage badge](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/stevenmburns/antenna_designer/python-coverage-comment-action-data/endpoint.json)](https://htmlpreview.github.io/?https://github.com/stevenmburns/antenna_designer/blob/python-coverage-comment-action-data/htmlcov/index.html)

Using this one will allow you to [customize](https://shields.io/endpoint) the look of your badge.
It won't work with private repositories. It won't be refreshed more than once per five minutes.

### [Shields.io](https://shields.io) Dynamic Badge

[![Coverage badge](https://img.shields.io/badge/dynamic/json?color=brightgreen&label=coverage&query=%24.message&url=https%3A%2F%2Fraw.githubusercontent.com%2Fstevenmburns%2Fantenna_designer%2Fpython-coverage-comment-action-data%2Fendpoint.json)](https://htmlpreview.github.io/?https://github.com/stevenmburns/antenna_designer/blob/python-coverage-comment-action-data/htmlcov/index.html)

This one will always be the same color. It won't work for private repos. I'm not even sure why we included it.

## What is that?

This branch is part of the
[python-coverage-comment-action](https://github.com/marketplace/actions/python-coverage-comment)
GitHub Action. All the files in this branch are automatically generated and may be
overwritten at any moment.