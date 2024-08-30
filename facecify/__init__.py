from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_login import LoginManager

app = Flask(__name__)
app.config['SECRET_KEY'] = 'e3cb708f2c7c257ae5b6b3ec4fe60e86'
app.config['SQLALCHEMY_DATABASE_URI'] = "postgresql://postgres:maloomhaina@localhost:5432/newDB"

bcrypt = Bcrypt(app)
db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'
login_manager.login_message_category = 'info'

from facecify import routes, models
