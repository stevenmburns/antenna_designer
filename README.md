# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/stevenmburns/antenna_designer/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                                                     |    Stmts |     Miss |   Cover |   Missing |
|------------------------------------------------------------------------- | -------: | -------: | ------: | --------: |
| src/antenna\_designer/\_\_init\_\_.py                                    |       13 |        2 |     85% |     50-51 |
| src/antenna\_designer/\_\_main\_\_.py                                    |        0 |        0 |    100% |           |
| src/antenna\_designer/builder.py                                         |      201 |        0 |    100% |           |
| src/antenna\_designer/cli.py                                             |      201 |       13 |     94% |85, 99, 113, 127-135, 334 |
| src/antenna\_designer/core.py                                            |        6 |        0 |    100% |           |
| src/antenna\_designer/designs/bowtie.py                                  |       26 |        0 |    100% |           |
| src/antenna\_designer/designs/bowtiearray1x2.py                          |        7 |        0 |    100% |           |
| src/antenna\_designer/designs/bowtiearray2x4.py                          |        7 |        0 |    100% |           |
| src/antenna\_designer/designs/bowtiearray.py                             |        7 |        0 |    100% |           |
| src/antenna\_designer/designs/delta\_looparray.py                        |       10 |        0 |    100% |           |
| src/antenna\_designer/designs/delta\_looparray\_1x4.py                   |        7 |        0 |    100% |           |
| src/antenna\_designer/designs/delta\_looparray\_1x4\_grouped.py          |        7 |        0 |    100% |           |
| src/antenna\_designer/designs/delta\_looparray\_2x2.py                   |        7 |        0 |    100% |           |
| src/antenna\_designer/designs/folded\_invveearray.py                     |        7 |        0 |    100% |           |
| src/antenna\_designer/designs/freq\_based/delta\_loop.py                 |       36 |        0 |    100% |           |
| src/antenna\_designer/designs/freq\_based/delta\_loop\_slanted.py        |       40 |        0 |    100% |           |
| src/antenna\_designer/designs/freq\_based/delta\_looparray\_with\_tls.py |       50 |        1 |     98% |        24 |
| src/antenna\_designer/designs/freq\_based/diamond\_loop.py               |       31 |        0 |    100% |           |
| src/antenna\_designer/designs/freq\_based/diamond\_loop\_turnstile.py    |       37 |        0 |    100% |           |
| src/antenna\_designer/designs/freq\_based/dipole\_turnstile.py           |       21 |        0 |    100% |           |
| src/antenna\_designer/designs/freq\_based/fandipole.py                   |       62 |        4 |     94% |123, 159-161 |
| src/antenna\_designer/designs/freq\_based/folded\_invvee.py              |       34 |        0 |    100% |           |
| src/antenna\_designer/designs/freq\_based/hentenna.py                    |       35 |        1 |     97% |        42 |
| src/antenna\_designer/designs/freq\_based/hentenna\_slant.py             |       43 |        0 |    100% |           |
| src/antenna\_designer/designs/freq\_based/hourglass.py                   |       33 |        0 |    100% |           |
| src/antenna\_designer/designs/freq\_based/hourglass\_slant.py            |       39 |        1 |     97% |        34 |
| src/antenna\_designer/designs/freq\_based/inv\_delta\_loop.py            |       30 |        0 |    100% |           |
| src/antenna\_designer/designs/freq\_based/invvee.py                      |       30 |        0 |    100% |           |
| src/antenna\_designer/designs/freq\_based/yagi.py                        |       38 |        0 |    100% |           |
| src/antenna\_designer/designs/hentenna\_array.py                         |        7 |        0 |    100% |           |
| src/antenna\_designer/designs/hexbeam.py                                 |       45 |        0 |    100% |           |
| src/antenna\_designer/designs/hourglass\_array.py                        |        7 |        0 |    100% |           |
| src/antenna\_designer/designs/invveearray.py                             |        8 |        0 |    100% |           |
| src/antenna\_designer/designs/moxon.py                                   |       33 |        0 |    100% |           |
| src/antenna\_designer/designs/moxonarray.py                              |        7 |        0 |    100% |           |
| src/antenna\_designer/designs/raised\_vertical.py                        |       26 |        0 |    100% |           |
| src/antenna\_designer/designs/twoband\_fan\_dipole.py                    |       79 |       12 |     85% |210, 322-337 |
| src/antenna\_designer/designs/vertical.py                                |       24 |        0 |    100% |           |
| src/antenna\_designer/designs/yagiarray.py                               |        7 |        0 |    100% |           |
| src/antenna\_designer/engine.py                                          |       13 |        2 |     85% |    42, 48 |
| src/antenna\_designer/engines/\_\_init\_\_.py                            |        3 |        0 |    100% |           |
| src/antenna\_designer/engines/pynec.py                                   |      104 |        7 |     93% |52, 71, 103, 105, 110, 141-143 |
| src/antenna\_designer/engines/pysim.py                                   |      154 |        4 |     97% |34, 72, 115, 259 |
| src/antenna\_designer/far\_field.py                                      |       94 |        0 |    100% |           |
| src/antenna\_designer/geometry.py                                        |      130 |        5 |     96% |54, 72, 83, 172, 245 |
| src/antenna\_designer/opt.py                                             |       39 |        1 |     97% |        69 |
| src/antenna\_designer/sim.py                                             |        2 |        0 |    100% |           |
| src/antenna\_designer/sweep.py                                           |      123 |        7 |     94% |   105-116 |
| src/antenna\_designer/transform.py                                       |       42 |        1 |     98% |        62 |
| **TOTAL**                                                                | **2012** |   **61** | **97%** |           |


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