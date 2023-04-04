"""
=============================================================================

    This module contains methods for handling database operations related to
    aqmlator.

=============================================================================

    Copyright 2022 ACK Cyfronet AGH. All Rights Reserved.

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

=============================================================================

    This work was supported by the EuroHPC PL project funded at the Smart Growth
    Operational Programme 2014-2020, Measure 4.2 under the grant agreement no.
    POIR.04.02.00-00-D014/20-00.

=============================================================================
"""

import aqmlator.database_connection as db
from subprocess import Popen
import sqlite3
import os

__author__ = "Tomasz Rybotycki"


def _dump_postgres_base(dump_file_name: str = "aqmlatorDump.sql") -> None:
    """
    Dumps postgres database to a file with given name.

    :param dump_file_name:
        Name of the file to dump the database into.
    """
    command: str = (
        "pg_dump --create --inserts -f "
        + dump_file_name
        + " -d "
        + db.get_database_url()
    )

    proc = Popen(command, shell=True)
    proc.wait()


def _parse_to_sqlite(sql_file: str = "aqmlatorDump.sql") -> None:
    """
    Parse the dumped postgres database into the sql format that can be used to
    initialize SQLite database.

    :param sql_file:
        A file containing dumped postgres database.
    """
    parsed_sql: str = ""

    with open(sql_file, "r") as f:
        while True:
            line: str = f.readline()

            # Finish parsing when needed.
            if line.__contains__("PostgreSQL database dump complete"):
                break

            # Keep only CREATE TABLE and INSERT commands.
            if not (line.__contains__("CREATE TABLE") or line.__contains__("INSERT")):
                continue

            # Remove `public.` from the database names.
            line = line.replace("public.", "")

            parsed_sql += line

            while not line.__contains__(";"):
                line = f.readline()
                line = line.replace("public.", "")
                parsed_sql += line

    with open("parsed_" + sql_file, "w") as f:
        f.write(parsed_sql)


def _initialize_sqlite_db(
    sql_file: str, sqlite_db_name: str = "aqmlatorSQLite.db"
) -> None:
    """
    Initializes the SQLite database from the given .sql file.

    :param sql_file:
        A file containing the `sql` script to initialize the SQLite database.
    :param sqlite_db_name:
        The name of the resultant SQLite database file.
    """

    database = sqlite3.connect(sqlite_db_name)

    with open(sql_file, "r") as f:
        database.executescript(f.read())

    database.close()


def export_data_to_sqlite_database(db_name: str = "aqmlatorSQLite.db") -> None:
    """
    Exports the aqmlator data to the SQLite database.

    :param db_name:
        Name of the resultant SQLite database.
    """
    sql_file_name: str = "temp.sql"
    _dump_postgres_base(sql_file_name)
    _parse_to_sqlite(sql_file_name)
    _initialize_sqlite_db("parsed_" + sql_file_name, db_name)

    # Remove sql files.
    os.remove(sql_file_name)
    os.remove("parsed_" + sql_file_name)
