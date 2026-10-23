import re

from flask import Blueprint, abort, flash, redirect, render_template, request, url_for

from ..conditions import CONDITIONS_BY_KEY
from ..extensions import db
from ..models import ContactMessage
from ..model_card import MODEL_CARD, line_chart

bp = Blueprint("pages", __name__)

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
CONTACT_TOPICS = {
    "general": "General question",
    "research": "Research collaboration",
    "bug": "Bug report",
    "other": "Something else",
}


@bp.route("/")
def home():
    return render_template("home.html", card=MODEL_CARD)


@bp.route("/conditions")
def conditions():
    return render_template("conditions.html")


@bp.route("/conditions/<key>")
def condition_detail(key):
    condition = CONDITIONS_BY_KEY.get(key)
    if condition is None:
        abort(404)
    return redirect(url_for("pages.conditions", _anchor=condition.key))


@bp.route("/model")
def model_card():
    history = MODEL_CARD["history"]
    charts = {
        "accuracy": line_chart(
            [("Training", history["accuracy"]), ("Validation", history["val_accuracy"])],
            y_min=0.4, y_max=1.0,
        ),
        "loss": line_chart(
            [("Training", history["loss"]), ("Validation", history["val_loss"])],
            y_min=0.0, y_max=2.0,
        ),
    }
    return render_template("model.html", card=MODEL_CARD, charts=charts)


@bp.route("/contact", methods=["GET", "POST"])
def contact():
    form = {"name": "", "email": "", "topic": "general", "message": ""}
    errors = {}
    if request.method == "POST":
        form = {k: request.form.get(k, "").strip() for k in form}
        if not form["name"]:
            errors["name"] = "Please enter a name."
        if not EMAIL_RE.match(form["email"]):
            errors["email"] = "Please enter a valid email address."
        if form["topic"] not in CONTACT_TOPICS:
            form["topic"] = "other"
        if len(form["message"]) < 10:
            errors["message"] = "The message needs at least 10 characters."

        if not errors:
            db.session.add(ContactMessage(**form))
            db.session.commit()
            flash("Thanks, the message is in. A reply usually follows within two working days.", "success")
            return redirect(url_for("pages.contact"))

    return render_template("contact.html", form=form, errors=errors, topics=CONTACT_TOPICS)


# Old URLs from the first version of the site.
@bp.route("/services")
def services():
    return redirect(url_for("pages.conditions"), code=301)


@bp.route("/about")
def about():
    return redirect(url_for("pages.model_card"), code=301)
