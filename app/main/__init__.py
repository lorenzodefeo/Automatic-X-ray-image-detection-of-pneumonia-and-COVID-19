from flask import Blueprint

main_blueprits = Blueprint("main", __name__)

import app.main.routes
