from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from pathlib import Path
import os

db = SQLAlchemy()


def create_app() -> Flask:
    """
    Application Factory.
    Chargement du modèle ML au démarrage de l'app.
    """
    app = Flask(__name__)

    # ── Configuration ──────────────────────────────────
    BASE_DIR = Path(__file__).parent.parent
    app.config["SECRET_KEY"]                  = os.environ.get("SECRET_KEY", "dev-only-change-me")
    app.config["SQLALCHEMY_DATABASE_URI"]     = (
        f"sqlite:///{BASE_DIR / 'data' / 'predictions.db'}"
    )
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

    # ── Initialisation extensions ───────────────────────
    # db = SQLAlchemy(app)
    db.init_app(app)

    with app.app_context():
        db.create_all()

        # Chargement du modèle ML au démarrage
        from app.models.fraud_detector import detector
        try:
            detector.load()
            app.logger.info(
                f"[OK] Modèle chargé : {detector.meta.get('model_name')}"
            )
        except FileNotFoundError:
            app.logger.warning(
                "[WARN] Modèle absent — lancez d'abord : python ml/train.py"
            )

    # ── Enregistrement Blueprint ────────────────────────
    from app.routes import main_bp
    app.register_blueprint(main_bp)

    return app