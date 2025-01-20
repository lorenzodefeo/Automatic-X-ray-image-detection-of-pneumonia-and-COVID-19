import os

APP_ROOT = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))


class Config:
    SECRET_KEY = os.environ.get("SECRET_KEY", "my_very_secret_key")
    USER_APP_NAME = "Automatic-X-ray-detection-of-pneumonia-and-COVID-19-App"


class DevelopmentConfig(Config):
    DEBUG = True
    # Altre configurazioni specifiche per lo sviluppo


class TestingConfig(Config):
    TESTING = True
    # Altre configurazioni specifiche per il testing


class ProductionConfig(Config):
    DEBUG = False
    # Altre configurazioni specifiche per la produzione


config_by_name = {
    'development': DevelopmentConfig,
    'testing': TestingConfig,
    'production': ProductionConfig,
}
