from urllib.parse import urlsplit

from flask import Blueprint, flash, redirect, render_template, request, url_for
from flask_login import current_user, login_required, login_user, logout_user

from ..extensions import db, login_manager
from ..models import User
from .pages import EMAIL_RE

bp = Blueprint("auth", __name__)

MIN_PASSWORD = 8


@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))


def _safe_next(default="analysis.dashboard"):
    target = request.args.get("next") or request.form.get("next")
    # Only allow relative paths so the redirect cannot leave the site.
    if target and not urlsplit(target).netloc and target.startswith("/") and not target.startswith("//"):
        return target
    return url_for(default)


@bp.route("/signin", methods=["GET", "POST"])
def signin():
    if current_user.is_authenticated:
        return redirect(url_for("analysis.dashboard"))

    email = ""
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        user = User.query.filter_by(email=email).first()
        if user and user.check_password(password):
            login_user(user, remember=bool(request.form.get("remember")))
            flash(f"Welcome back, {user.first_name}.", "success")
            return redirect(_safe_next())
        flash("That email and password combination does not match an account.", "error")

    return render_template("auth/signin.html", email=email)


@bp.route("/signup", methods=["GET", "POST"])
def signup():
    if current_user.is_authenticated:
        return redirect(url_for("analysis.dashboard"))

    form = {"name": "", "email": ""}
    errors = {}
    if request.method == "POST":
        form = {
            "name": request.form.get("name", "").strip(),
            "email": request.form.get("email", "").strip().lower(),
        }
        password = request.form.get("password", "")
        confirm = request.form.get("confirm", "")

        if len(form["name"]) < 2:
            errors["name"] = "Please enter a name."
        if not EMAIL_RE.match(form["email"]):
            errors["email"] = "Please enter a valid email address."
        elif User.query.filter_by(email=form["email"]).first():
            errors["email"] = "An account with this email already exists."
        if len(password) < MIN_PASSWORD:
            errors["password"] = f"Use at least {MIN_PASSWORD} characters."
        elif password != confirm:
            errors["confirm"] = "The passwords do not match."

        if not errors:
            user = User(name=form["name"], email=form["email"])
            user.set_password(password)
            db.session.add(user)
            db.session.commit()
            login_user(user)
            flash("Account created. Analyses run while signed in are now saved to the history.", "success")
            return redirect(_safe_next())

    return render_template("auth/signup.html", form=form, errors=errors, min_password=MIN_PASSWORD)


@bp.route("/signout", methods=["POST"])
@login_required
def signout():
    logout_user()
    flash("Signed out.", "info")
    return redirect(url_for("pages.home"))
