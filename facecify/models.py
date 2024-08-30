from flask_login import UserMixin
from facecify import db, login_manager

@login_manager.user_loader
def load_user(user_id):
    return Faculty.query.get(user_id)

# Association tables
faculty_classes = db.Table('faculty_classes',
    db.Column('faculty_id', db.Integer, db.ForeignKey('faculty.id'), primary_key=True),
    db.Column('class_id', db.Integer, db.ForeignKey('class.id'), primary_key=True)
)

subject_classes = db.Table('subject_classes',
    db.Column('subject_id', db.Integer, db.ForeignKey('subject.id'), primary_key=True),
    db.Column('class_id', db.Integer, db.ForeignKey('class.id'), primary_key=True)
)


class Faculty(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)
    image_file = db.Column(db.String(20), nullable=False, default='default.jpg')
    is_admin = db.Column(db.Boolean, default=False)
    classes = db.relationship('Class', secondary=faculty_classes, lazy='subquery',
        backref=db.backref('faculties', lazy=True))

class Class(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    class_code = db.Column(db.String(10), unique=True, nullable=False)
    course = db.Column(db.String(100), nullable=False)
    section = db.Column(db.String(10), nullable=False)
    branch = db.Column(db.String(50), nullable=False)
    semester = db.Column(db.String(10), nullable=False)
    subjects = db.relationship('Subject', secondary=subject_classes, backref=db.backref('classes', lazy=True))

class Subject(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(30), unique=True, nullable=False)
    branch = db.Column(db.String(50), nullable=False)
    subject_code = db.Column(db.String(10), nullable=True)
    class_id = db.Column(db.Integer, db.ForeignKey('class.id'), nullable=True)

class Student(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(200), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    enrollment_number = db.Column(db.String(20), unique=True, nullable=False)
    video = db.Column(db.String(120), nullable=True)  # URL or path to the video file
    class_id = db.Column(db.Integer, db.ForeignKey('class.id'), nullable=True)
    is_active=db.Column(db.Boolean, default= False)
    class_ = db.relationship('Class', backref=db.backref('students', lazy=True))
    
    def __repr__(self):
        return f"Student('{self.name}', '{self.email}', '{self.enrollment_number}', '{self.class_.class_code}')"

