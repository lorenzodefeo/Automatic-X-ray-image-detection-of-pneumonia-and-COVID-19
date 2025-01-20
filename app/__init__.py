import os

from flask import Flask
from .config import Config
from app.errors.routes import error_401, error_403, error_404, error_500


def create_app(config_name=None):
    if config_name is None:
        config_name = os.getenv('FLASK_ENV', 'development')

    app = Flask(__name__)
    app.config.from_object(Config)

    # Initialization of blueprits
    # I Blueprints in Flask sono uno strumento potente per organizzare, riutilizzare e
    # mantenere il codice delle applicazioni web in modo efficiente, rendendo più semplice la gestione di
    # applicazioni più complesse e scalabili

    from app.main import main_blueprits
    app.register_blueprint(main_blueprits)

    # Error handlers
    app.register_error_handler(404, error_404)
    app.register_error_handler(403, error_403)
    app.register_error_handler(401, error_401)
    app.register_error_handler(500, error_500)

    return app
