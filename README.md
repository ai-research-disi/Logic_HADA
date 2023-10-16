# Logic HADA

## How to run

1. Install the requirements in a `python 3.7` environment
    ```
    pip install -r requirements.txt
    ```
2. Run the script
    ```
    python run.py
    ```

## How to enforce the logic constraints

GridREx extracts IF_THEN rules: IF is composed by AND of range constraints and THEN is an algebraic equation.
For example:

```
IF x1 in (0.99, 20.79) AND x2 in (33.1, 78.4)
THEN y == 66.20 + 4.02 * x1 + 3.48 * x2
```

We enforce the logic rules as indicator constraints:

```
# IF statement
    if z1 == 1 then x1 in (0.99, 20.79)
    if z2 == 1 then x2 in (33.1, 78.4)
# Link between IF and THEN statement
    if z_THEN == 1 then z1 + z2 == 2  # i.e., both z1 and z2 are equal to 1
# THEN statement    
    if z_THEN == 1 then y = 66.20 + 4.02 * x1 + 3.48 * x2 
```

Since GridREx splits the input space in hypercubes, we have multiple logic rules associated to a single ML model,
but we want one and only one of them to be true per time. Thus, we force the sum of the THEN variables (there is one for
each rule) to be equal to 1.


## Which are the **variables** involved in the optimization process?

* Binary variable to signify the algorithm `ALG`
    * `b_{ALG}` in {0,1}

* Binary variables to express the i-th logic rule
    * `{ALG}_LogRul_{i}_IF_z{j}` in {0,1}
    * `{ALG}_LogRul_{i}_THEN_z{j}` in {0,1}

* Continuous variables for the ML model targets (`'memAvg(MB)'`, `'time(sec)'`, `'sol(keuro)'`) with their upper and
  lower bounds
    * `y_{ALG}_{VAR}` in [{VAR}_lb, {VAR}_ub]

* Continuous variables for the input features (`'PV_mean'`, `'PV_std'`, `'Load_mean'`, `'Load_std'`), which form the
  instance description, with their upper and lower bounds
    * `y_{VAR}` in [{VAR}_lb, {VAR}_ub]

* Continuous variables for the algorithm hyperparameters (`nTraces` or  `'nScenarios'`) with their upper and lower
  bounds
    * `y_{VAR}` in [{VAR}_lb, {VAR}_ub]



## Which are the **constraints** involved in the optimization process?

1. The sum of the binary algorithm variables must be equal to 1, so only 1 algorithm can be chosen (ANT or CONT).
2. The logic rules (GridREx rules) forced as indicators constraints.
3. The sum over i of `{ALG}_LogRul_{i}_THEN_z{j}` must be equal to 1, so only 1 logic rule can be true at each time.
4. The user-defined constraints.

# Author 
* **Eleonora Misino** ([eleonora.misino2@unibo.it](mailto:eleonora.misino2@unibo.it))
