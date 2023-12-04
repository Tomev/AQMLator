# AQMLator

A package for auto (quantum machine learning)-izing your experiments!

## Requirements

Python version 3.11 is required. Necessary packages are provided in the respective
requirements*.txt files.

### Database

The Postgres database is used to store the Optuna trials data. It has to be 
installed prior to running the package, so that the `psycopg2` package can be
installed properly.

### Documentation

The documentation is built using Sphinx. Building PDF version of the documentation
requires `latex` distribution (we used `miktex`) and `perl` (we used `strawberry perl`).
To generate the pdf version of documentation, run

`make latexpdf`

and to generate the html version of documentation, run

`make html`

Both command should be run from the `docs` directory.

## Installation

The package is available on pip, and can be installed using

`pip install aqmlator`

To install the package from the sources, run

`pip install .`

To develop the package, run

`pip install -e .`

To install the packages required for development, run

`pip install -r requirements -r requirements-dev.txt`

## Setup



## Access

To access the Optuna trials data use 
[optuna-dashboard](https://github.com/optuna/optuna-dashboard)
application. By default, it can be run using the following command

`optuna-dashboard postgresql://user:password@localhost/mydb`

while the (PostgreSQL) database is running.

Alternatively, one can use `aqmlator.database.export_data_to_sqlite_database` to export
the data to the SQLite database, and handle it as one pleases.
