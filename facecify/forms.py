import datetime
from flask_login import current_user
from flask_wtf import FlaskForm
from flask_wtf.file import FileField,FileAllowed
from wtforms import BooleanField, DateField, IntegerField, SelectField, StringField, PasswordField,SubmitField, ValidationError
from wtforms.validators import DataRequired, Length, Email ,EqualTo, Optional,InputRequired
from facecify.models import Class, Faculty, Subject
from flask_login import current_user
from datetime import date
from facecify.models import Student,Subject,Class

class RegistrationForm(FlaskForm):
    username= StringField('Username',
                           validators=[DataRequired(), Length(min=2, max=20)])
    email=StringField('Email', 
                      validators=[DataRequired(), Email()])
    password= PasswordField('Password',
                            validators=[DataRequired()])
    confirm_password= PasswordField('Confirm Password',
                            validators=[DataRequired(), EqualTo('password')])
    submit= SubmitField('Sign Up')

    def validate_username(self,username):
        faculty = Faculty.query.filter_by(username=username.data).first()
        if faculty:
            raise ValidationError('This username is already Taken, try another username')
        
    def validate_email(self,email):
        faculty = Faculty.query.filter_by(email=email.data).first()
        if faculty:
            raise ValidationError('This email is already in use, try to log in')

class LoginForm(FlaskForm):
    email=StringField('Email', 
                      validators=[DataRequired(), Email()])
    password= PasswordField('Password',
                            validators=[DataRequired()])
    submit= SubmitField('Login')
    remember=BooleanField('Remember')

class UpdateAccountForm(FlaskForm):
    username = StringField('Username', 
                           validators=[DataRequired(), Length(min=2, max=20)])
    email = StringField('Email', 
                        validators=[DataRequired(), Email()])
    picture = FileField('Update Profile Picture', validators=[FileAllowed(['jpg', 'png'])])
    class_ = SelectField('Add Class', choices=[], validators=[Optional()])
    submit = SubmitField('Update')

    def __init__(self, *args, **kwargs):
        super(UpdateAccountForm, self).__init__(*args, **kwargs)
        # Add 'None' option to choices
        self.class_.choices = [(0, 'None')] + [(class_.id, class_.class_code) for class_ in Class.query.all()]

    def validate_username(self, username):
        if username.data != current_user.username:
            user = Faculty.query.filter_by(username=username.data).first()
            if user:
                raise ValidationError('That username is taken. Please choose a different one.')

    def validate_email(self, email):
        if email.data != current_user.email:
            user = Faculty.query.filter_by(email=email.data).first()
            if user:
                raise ValidationError('That email is taken. Please choose a different one.')

class AddClassForm(FlaskForm):
    class_code = StringField('Class Code', validators=[DataRequired(), Length(max=20)])
    course = StringField('Course', validators=[DataRequired(), Length(max=100)])
    section = StringField('Section', validators=[DataRequired(), Length(max=10)])
    def validate_branch(form, field):
        if field.data == '':
            raise ValidationError('Please select a branch.')

    branch = SelectField('Branch', choices=[
        ('', 'None'),  # Default selection
        ('B.Tech', 'B.Tech'),
        ('BBA', 'BBA'),
        ('B.Des', 'B.Des'),
        ('M.Tech', 'M.Tech'),
        ('MBA', 'MBA'),
        ('M.Des', 'M.Des'),
        ('B.Sc', 'B.Sc'),
        ('B.Com', 'B.Com'),
        ('M.Sc', 'M.Sc'),
        ('M.Com', 'M.Com')
    ], validators=[validate_branch])

    semester = StringField('Semester', validators=[DataRequired(), Length(max=10)])
    submit = SubmitField('Add Class')

    def validate_class_code(self, new_class_code):
        new_class = Class.query.filter_by(class_code=new_class_code.data).first()
        if new_class:
            raise ValidationError('That class_code is taken. Please choose a different one.')

class AddSubjectForm(FlaskForm):
    name = StringField('Subject Name', validators=[DataRequired(), Length(max=30)])
    subject_code = StringField('Subject Code', validators=[Length(max=10)])
    def validate_branch(form, field):
        if field.data == '':
            raise ValidationError('Please select a branch.')

    branch = SelectField('Branch', choices=[
        ('', 'None'),  # Default selection
        ('B.Tech', 'B.Tech'),
        ('BBA', 'BBA'),
        ('B.Des', 'B.Des'),
        ('M.Tech', 'M.Tech'),
        ('MBA', 'MBA'),
        ('M.Des', 'M.Des'),
        ('B.Sc', 'B.Sc'),
        ('B.Com', 'B.Com'),
        ('M.Sc', 'M.Sc'),
        ('M.Com', 'M.Com')
    ], validators=[validate_branch])
    submit = SubmitField('Add Subject')

    def validate_name(self, name):
        sub = Subject.query.filter_by(name=name.data).first()
        if sub:
            raise ValidationError('That Subject already exists. Please choose a different one.')
    def validate_subject(self, subject_code):
        sub = Subject.query.filter_by(subject_code=subject_code.data).first()
        if sub:
            raise ValidationError('That Subject Code already exists. Please choose a different one.')
        
class AssignSubjectForm(FlaskForm):
    class_ = SelectField('Class', coerce=int, validators=[DataRequired()])
    subject = SelectField('Subject', coerce=int, validators=[DataRequired()])
    submit = SubmitField('Assign')
    
class MarkAttendance(FlaskForm):
    class_ = SelectField('Class', coerce=int, validators=[DataRequired()])
    subject = SelectField('Subject', coerce=int, validators=[DataRequired()])
    day= date.today()
    submit = SubmitField('Assign')


class StudentRegistrationForm(FlaskForm):
    name = StringField('Name', validators=[DataRequired(), Length(max=100)])
    email = StringField('Email', validators=[DataRequired(), Email()])
    enrollment_number = StringField('Enrollment Number', validators=[DataRequired(), Length(max=20)])
    class_ = SelectField('Class', coerce=int, validators=[DataRequired()])
    password = PasswordField('Password', validators=[DataRequired()])
    confirm_password = PasswordField('Confirm Password', validators=[DataRequired(), EqualTo('password')])
    submit = SubmitField('Register')

    def validate_enrollment_number(self, enrollment_number):
        if not enrollment_number.data.startswith('IU'):
            raise ValidationError('Enrollment number must start with "IU"')
    # def validate_email(self, email):
    #     if not email.data.endswith('.indusuni.ac.in'):
    #         raise ValidationError('Use Indus Email ID')
    def validate_class_(self, class_):
        if class_.data == 'None':
            raise ValidationError('Please select a valid class')
        
class OTPVerificationForm(FlaskForm):
    otp = StringField('OTP', validators=[DataRequired()])
    submit = SubmitField('Verify OTP')

class VideoUploadForm(FlaskForm):
    video = FileField('Upload Video', validators=[DataRequired()])
    submit = SubmitField('Submit Video')
        
class ClassSelectionForm(FlaskForm):
    class_ = SelectField('Class', choices=[], validators=[DataRequired()])

class ClassSubjectSelectionForm(FlaskForm):
    class_ = SelectField('Class', choices=[], validators=[DataRequired()])
    subject = SelectField('Subject', choices=[(None, 'None')], validators=[DataRequired()])
    date = DateField('Date', default=datetime.date.today, format='%Y-%m-%d', validators=[DataRequired()])
    submit = SubmitField('Submit')
        