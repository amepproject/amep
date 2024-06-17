# Refactoring tests

The goal is to make the tests not dependant on downloads. So we wanna use our provided example data.
Also we wann increase code coverage and decrease multiple executions.
In order to do this I list all test files here as sections

| test script       | depends on downloads  | unnecessary duplications  |
| ----------------- | --------------------- | ------------------------- |
| base              |          ✔            |            ✔              |
| continuum         |          ✔            |            ✔              |
| evaluate          |          ✔            |            ✔              |
| functions         |          ✔            |            ✔              |
| load              |          ✔            |            ✔              |
| order             |          ✔            |            ✔              |
| particle methods  |          ✔            |            ✔              |
| pbc               |          ✔            |            ✔              |
| plot              |          ✔            |            ✔              |
| reader            |          ✔            |            ✔              |
| spatialcor        |          ✔            |            ✔              |
| timecor           |          ✔            |            ✔              |
| trajectory        |          ✔            |            ✔              |
