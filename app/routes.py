from flask import (
    Blueprint, request, jsonify,
    render_template, redirect, url_for, abort
)
from datetime import datetime
import json

from app import db
from app.models.fraud_detector import detector

main_bp = Blueprint("main", __name__)


# ── Modèle SQLAlchemy ─────────────────────────────────
class Prediction(db.Model):
    """
    définition d'un modèle SQLAlchemy.
    """
    __tablename__ = "predictions"

    id                = db.Column(db.Integer,    primary_key=True)
    timestamp         = db.Column(db.DateTime,   default=datetime.utcnow,
                                                  nullable=False)
    amount            = db.Column(db.Float,      nullable=False)
    time_seconds      = db.Column(db.Float,      nullable=False)
    fraud_probability = db.Column(db.Float,      nullable=False)
    is_fraud          = db.Column(db.Boolean,    nullable=False)
    risk_level        = db.Column(db.String(10), nullable=False)
    input_features    = db.Column(db.Text,       nullable=True)
    true_label        = db.Column(db.Integer,    nullable=True)

    def to_dict(self) -> dict:
        """
        retourner JSON depuis un modèle.
        """
        return {
            "id":                self.id,
            "timestamp":         self.timestamp.isoformat(),
            "amount":            self.amount,
            "time_seconds":      self.time_seconds,
            "fraud_probability": round(self.fraud_probability, 4),
            "is_fraud":          self.is_fraud,
            "risk_level":        self.risk_level,
            "true_label":        self.true_label,
        }


# ─────────────────────────────────────────────────────
# GET /  — Formulaire de saisie
# render_template + variables Jinja2
# ─────────────────────────────────────────────────────
@main_bp.route("/", methods=["GET"])
def index():
    """
    Affiche le formulaire de saisie (index.html).
    @app.route("/")
    render_template("index.html", variable=valeur)
    """
    feature_names = detector.feature_names if detector.is_loaded else []
    v_features    = [f for f in feature_names if f.startswith("V")]

    # return render_template('page.html', var=val)
    return render_template(
        "index.html",
        v_features   = v_features,
        model_loaded = detector.is_loaded,
        model_name   = detector.meta.get("model_name", "N/A")
    )


# ─────────────────────────────────────────────────────
# POST /predict  — Prédiction (endpoint principal)
# methods=['GET', 'POST']
# request.get_json()
# jsonify + codes HTTP
# ─────────────────────────────────────────────────────
@main_bp.route("/predict", methods=["POST"])
def predict():
    """
    Reçoit les features d'une transaction.
    Retourne la probabilité de fraude en JSON.
    
    if not request.is_json:
        return jsonify({'erreur': '...'}), 400
    """
    # Vérification modèle chargé
    if not detector.is_loaded:
        # codes HTTP 503
        return jsonify({
            "error":   "Modèle non chargé",
            "conseil": "Lancez d'abord : python ml/train.py"
        }), 503

    # request.form
    # request.get_json()
    if request.is_json:
        data = request.get_json(force=True)
    else:
        data = request.form.to_dict()

    # Validation des champs requis
    # abort(400) pour données invalides
    missing = [k for k in ["Amount", "Time"] if k not in data]
    if missing:
        return jsonify({
            "error":   "Champs manquants",
            "missing": missing
        }), 400

    try:
        # Conversion en float
        input_data = {
            k: float(v)
            for k, v in data.items()
            if k not in ("true_label", "submit")
        }

        # Prédiction ML
        result = detector.predict(input_data)

        # On récupère la probabilité que le detector a déjà calculée
        prob = result["fraud_probability"]

        # On impose notre nouveau seuil de 0.35
        new_is_fraud = True if prob >= 0.35 else False
        
        # On met à jour le résultat avec notre décision à 0.35
        result["is_fraud"] = new_is_fraud
        result["risk_level"] = "HIGH" if prob >= 0.7 else ("MEDIUM" if prob >= 0.35 else "LOW")
        
        # Journalisation SQLite
        # db.session.add() + db.session.commit()
        pred_record = Prediction(
            amount            = input_data.get("Amount", 0.0),
            time_seconds      = input_data.get("Time",   0.0),
            fraud_probability = result["fraud_probability"],
            is_fraud          = new_is_fraud,
            risk_level        = result["risk_level"],
            input_features    = json.dumps(input_data),
            true_label        = int(data["true_label"])
                                if data.get("true_label", "") != ""
                                else None,
        )
        db.session.add(pred_record)
        db.session.commit()

        result["prediction_id"] = pred_record.id
        result["timestamp"]     = pred_record.timestamp.isoformat()

        # redirect(url_for('...'))
        if request.is_json:
            # return jsonify(...), 200
            return jsonify(result), 200
        else:
            # Soumis depuis le formulaire HTML → result.html
            return redirect(
                url_for("main.result_page",
                        prediction_id=pred_record.id)
            )

    except ValueError as e:
        # codes HTTP 400
        return jsonify({"error": f"Valeur invalide : {str(e)}"}), 400
    except Exception as e:
        # codes HTTP 500
        return jsonify({"error": f"Erreur serveur : {str(e)}"}), 500


# ─────────────────────────────────────────────────────
# GET /predict/<id>  — Affichage résultat (result.html)
# paramètre dynamique <int:id>
# render_template
# ─────────────────────────────────────────────────────
@main_bp.route("/predict/<int:prediction_id>", methods=["GET"])
def result_page(prediction_id: int):
    """
    Affiche le résultat d'une prédiction — result.html.
    @app.route('/produit/<int:id>')
    abort(404) si introuvable
    """
    # Utilisateur.query.get(1)
    pred = db.session.get(Prediction, prediction_id)
    if pred is None:
        # abort(404)
        abort(404)

    input_feats = json.loads(pred.input_features) \
                  if pred.input_features else {}

    # render_template avec variables
    return render_template(
        "result.html",
        pred           = pred,
        input_features = input_feats
    )


# ─────────────────────────────────────────────────────
# GET /stats  — Statistiques du modèle (JSON)
# return jsonify(...), 200
# ─────────────────────────────────────────────────────
@main_bp.route("/stats", methods=["GET"])
def stats():
    """
    Statistiques du modèle ML + métriques des prédictions.
    endpoint GET → jsonify → 200
    """
    # Utilisateur.query.all() / .count()
    total       = Prediction.query.count()
    total_fraud = Prediction.query.filter_by(is_fraud=True).count()

    labeled = Prediction.query.filter(
        Prediction.true_label.isnot(None)
    ).all()

    tp = sum(1 for p in labeled if     p.is_fraud and p.true_label == 1)
    tn = sum(1 for p in labeled if not p.is_fraud and p.true_label == 0)
    fp = sum(1 for p in labeled if     p.is_fraud and p.true_label == 0)
    fn = sum(1 for p in labeled if not p.is_fraud and p.true_label == 1)

    # return jsonify({...}), 200
    return jsonify({
        "model": detector.get_stats(),
        "predictions": {
            "total":         total,
            "fraud_count":   total_fraud,
            "legit_count":   total - total_fraud,
            "fraud_rate":    round(total_fraud / total, 4) if total else 0,
            "labeled_count": len(labeled),
            "confusion": {
                "TP": tp, "TN": tn,
                "FP": fp, "FN": fn
            }
        }
    }), 200


# ─────────────────────────────────────────────────────
# GET /history  — 50 dernières prédictions (JSON)
# request.args.get()
# .query.order_by().limit().all()
# ─────────────────────────────────────────────────────
@main_bp.route("/history", methods=["GET"])
def history():
    """
    Liste les 50 dernières prédictions en JSON.
    request.args.get('limit', 50, type=int)
    .query.order_by(...).limit(...).all()
    """
    # request.args.get('page', 1, type=int)
    limit = min(
        request.args.get("limit", 50, type=int),
        200
    )

    # requête avancée avec order_by + limit
    predictions = (
        Prediction.query
        .order_by(Prediction.timestamp.desc())
        .limit(limit)
        .all()
    )

    # return jsonify({...}), 200
    return jsonify({
        "count":       len(predictions),
        "predictions": [p.to_dict() for p in predictions]
    }), 200


# ─────────────────────────────────────────────────────
# GET /admin  — Dashboard FP/FN (JSON)
# endpoint GET → jsonify → 200
# ─────────────────────────────────────────────────────
@main_bp.route("/admin", methods=["GET"])
def admin():
    """
    Dashboard admin — faux positifs / faux négatifs.
    return jsonify({...}), 200
    """
    labeled = Prediction.query.filter(
        Prediction.true_label.isnot(None)
    ).order_by(Prediction.timestamp.desc()).all()

    fp_list = [p for p in labeled if     p.is_fraud and p.true_label == 0]
    fn_list = [p for p in labeled if not p.is_fraud and p.true_label == 1]
    tp_list = [p for p in labeled if     p.is_fraud and p.true_label == 1]
    tn_list = [p for p in labeled if not p.is_fraud and p.true_label == 0]

    total       = Prediction.query.count()
    fraud_count = Prediction.query.filter_by(is_fraud=True).count()

    return jsonify({
        "model_name":    detector.meta.get("model_name", "N/A"),
        "model_metrics": detector.meta.get("metrics", {}),
        "summary": {
            "total":           total,
            "fraud_count":     fraud_count,
            "legit_count":     total - fraud_count,
            "labeled_count":   len(labeled),
            "unlabeled_count": Prediction.query.filter(
                                   Prediction.true_label.is_(None)
                               ).count(),
        },
        "confusion": {
            "TP": len(tp_list),
            "TN": len(tn_list),
            "FP": len(fp_list),
            "FN": len(fn_list),
        },
        "false_positives": [p.to_dict() for p in fp_list[:20]],
        "false_negatives": [p.to_dict() for p in fn_list[:20]],
    }), 200


# ─────────────────────────────────────────────────────
# Gestion des erreurs HTTP
# abort(404), abort(400)
# codes 400, 404, 500
# ─────────────────────────────────────────────────────
@main_bp.app_errorhandler(404)
def not_found(e):
    # 404 Not Found
    return jsonify({"error": "Ressource introuvable"}), 404


@main_bp.app_errorhandler(400)
def bad_request(e):
    # 400 Bad Request
    return jsonify({"error": "Requête invalide"}), 400


@main_bp.app_errorhandler(500)
def server_error(e):
    # 500 Server Error
    return jsonify({"error": "Erreur serveur interne"}), 500