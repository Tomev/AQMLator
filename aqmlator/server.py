"""
=============================================================================

    This module contains a server implementation for the AQMLator application.

=============================================================================

    Copyright 2023 ACK Cyfronet AGH. All Rights Reserved.

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

__author__ = "Tomasz Rybotycki"

from typing import Dict

import uvicorn
from fastapi import FastAPI, Response

app: FastAPI = FastAPI()

status_data: Dict[str, str] = {}

port: str = "8000"
address: str = "127.0.0.1"
status_update_endpoint: str = f"http://{address}:{port}/update_status"


@app.get("/status")
def status() -> Response:
    content: str = ""

    if not status_data:
        content += "No running tuners!"
    else:
        for status_id, tuning_status in status_data.items():
            content += f"Tuner {status_id}: {tuning_status}\n"

    return Response(status_code=200, content=content)


@app.post("/update_status")
async def get_data(data: Dict[str, str]) -> Response:
    for k in data:
        if data[k] == "Delete":
            status_data.pop(k)
        else:
            status_data[k] = data[k]
    return Response(status_code=200, content="Status updated!")


# app.add_api_route("/status", status, methods=["GET"])

if __name__ == "__main__":
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True)
