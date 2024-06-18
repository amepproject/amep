# Refactoring tests

The goal is to make the tests not dependant on downloads. So we wanna use our provided example data.
Also we wann increase code coverage and decrease multiple executions.
In order to do this I list all test files here as sections

| test script       | depends on downloads/data gen | unnecessary duplications  |
| ----------------- | ----------------------------- | ------------------------- |
| base              |          X                    |            X              |
| continuum         |          ✔                    |            ✔              |
| evaluate          |          ✔                    |            ✔              |
| functions         |          ✔                    |            ✔              |
| load              |          ✔                    |            ✔              |
| order             |          ✔                    |            ✔              |
| particle methods  |          ✔                    |            ✔              |
| pbc               |          ✔                    |            ✔              |
| plot              |          ✔                    |            ✔              |
| reader            |          ✔                    |            ✔              |
| spatialcor        |          ✔                    |            ✔              |
| timecor           |          ✔                    |            ✔              |
| trajectory        |          ✔                    |            ✔              |

## base

generiert selbst testdaten. Ein paar doppelte ausführungen sind auch dabei.
Wichtig sind:

    1. Umschreiben auf beispieldaten.
    2. Dopplungen in der testcoverage entfernen.
    3. Coverage in den Tests auf 1 erhöhen.

Getestet wird:

    1. Konstruktion von Basisstrukturen
        a. BaseField
        b. BaseTrajectory
        c. BaseFunction


