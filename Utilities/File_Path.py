"""
title: Files
author: open-webui
author_url: https://github.com/open-webui
funding_url: https://github.com/open-webui
version: 0.1.0
"""

import os
import requests
from datetime import datetime
from typing import List, Optional

from open_webui.config import UPLOAD_DIR
from pydantic import BaseModel, Field
import json


class Filter:
    class Valves(BaseModel):
        priority: int = Field(
            default=0, description="Priority level for the filter operations."
        )
        max_turns: int = Field(
            default=50, description="Maximum allowable conversation turns for a user."
        )
        pass

    class UserValves(BaseModel):
        max_turns: int = Field(
            default=50, description="Maximum allowable conversation turns for a user."
        )
        pass

    def get_file_content(self, body: dict) -> str:
        """
        Construye y retorna la ruta completa del fichero a partir de la información del JSON.
        Por ejemplo, con el siguiente JSON:
        {
          "id": "141caa11-dddc-4c06-ae08-830889ccc3ab",
          "filename": "ventas_campero (4).csv",
          ...
        }

        """
        try:
            file_info = body["files"][0]["file"]
            file_id = file_info.get("id")
            file_name = file_info.get("filename")
            if not file_id or not file_name:
                return ""

            # Se construye la ruta completa uniendo el directorio base, el id y el nombre del fichero.
            print(UPLOAD_DIR)
            full_path = os.path.join(UPLOAD_DIR, f"{file_id}_{file_name}")
            print("Ruta completa generada:", full_path)
            return full_path
        except (KeyError, IndexError) as e:
            return ""

    def __init__(self):
        # If set to true it will prevent default RAG pipeline
        # self.file_handler = True
        # self.citation = True
        self.valves = self.Valves()
        pass

    async def on_startup(self):
        # This function is called when the server is started.
        print(f"on_startup:{__name__}")
        pass

    async def on_shutdown(self):
        # This function is called when the server is stopped.
        print(f"on_shutdown:{__name__}")
        pass

    async def on_valves_updated(self):
        # This function is called when the valves are updated.
        pass

    async def inlet(self, body: dict, __user__: Optional[dict] = None) -> dict:
        # Se obtiene la ruta completa del fichero.
        file_path = self.get_file_content(body)

        if file_path:
            # Se construye el mensaje del sistema con el aviso y la ruta del fichero.
            system_msg = f"Para procesarlo con python puede acceder directamente al fichero: {file_path}"
            print("System Mensaje: \n\n\n\n")
            print(system_msg)
            context = {"role": "system", "content": system_msg}

            # Imprime todo el contenido de 'files' en formato JSON.
            # print("Contenido de 'body[\"files\"]' en JSON:")
            # print(json.dumps(body.get("files"), ensure_ascii=False, indent=2))

            # Se comprueba si ya existe un mensaje de sistema en el array de mensajes.
            if not body["messages"] or body["messages"][0].get("role") != "system":
                body["messages"].insert(0, context)
            else:
                body["messages"][0]["content"] = system_msg

            # Se elimina la clave 'files' del body ya que no se usará más adelante.
            # body["files"] = None

            # print(f"inlet:{__name__}")
            # print(f"inlet:body:{body}")
            # print(f"inlet:user:{__user__}")
            # print(body["messages"])

        return body
