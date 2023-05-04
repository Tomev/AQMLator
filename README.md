# AQMLator

A package for auto (quantum machine learning)-izing your experiments!

## Requirements

Python version 3.11 is required. Necessary packages are provided in the respective
requirements*.txt files.

## Access

To access the Optuna trials data use 
[optuna-dashboard](https://github.com/optuna/optuna-dashboard)
application. By default, it can be run using the following command

`optuna-dashboard postgresql://user:password@localhost/mydb`

while the (PostgreSQL) database is running.

Alternatively, one can use `aqmlator.database.export_data_to_sqlite_database` to export
the data to the SQLite database, and handle it as one pleases.
