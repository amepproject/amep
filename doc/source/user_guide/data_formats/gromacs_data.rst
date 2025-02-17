GROMACS data
------------

Our newest addition to **AMEP** is the possibilty to read `GROMACS <https://www.gromacs.org/>`_
simulation data. GROMACS (Groningen Machine for Chemical Simulations)
is commonly used for molecular dynamics simulations. Although not
active, the dynamics and behaviour on a molecular level
can give fascinating new insights into the whole realm of soft matter.

We use the python package ``chemfiles`` (`<https://chemfiles.org/>`_) to read
the binary trajectory and topology/run input files.

.. note::
    Currently, chemfiles does not read forces from trajectory files. 
    `<https://github.com/chemfiles/chemfiles/issues/496>`_

    As soon as this is supported by chemfiles this feature will be included
    **AMEP**.

**AMEP** also includes a basic mechanism to identify molecules,
making molecule-based analysis more accessible than using GROMACS
index files.