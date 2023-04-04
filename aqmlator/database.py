"""
=============================================================================

    This module contains ...

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

__author__ = "Tomasz Rybotycki"


def dump(dumpName: str = "aqmlatorDump.sql") -> None:
    """

    """
    command: str = "pg_dump --create --inserts -f " + dumpName + " -d "\
                   + db.get_database_url()


    proc = Popen(command, shell=True)
    proc.wait()

def parse_to_sqlite(sql: str) -> None:
    """

    """
    parsed_sql: str = ""

    with open(sql, "r") as f:
        while True:
            line: str = f.readline()

            if line.__contains__("PostgreSQL database dump complete"):
                break

            if not (line.__contains__("CREATE TABLE") or line.__contains__("INSERT")):
                continue

            line = line.replace("public.", "")

            parsed_sql += line

            while not line.__contains__(";"):
                line = f.readline()
                line = line.replace("public.", "")
                parsed_sql += line

    with open("parsed_aqmlatorDump.sql", "w") as f:
        f.write(parsed_sql)

def create_sqlite_db(sql_flie: str, sqlite_db_name: str = "aqmlatorSQLite.db") -> None:
    """

    """
    db = sqlite3.connect(sqlite_db_name)

    with open(sql_flie, "r") as f:
        db.executescript(f.read())

    db.close()
