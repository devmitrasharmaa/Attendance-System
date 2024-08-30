import csv
import os
import pickle
import random
import cv2 as cv
from keras_facenet import FaceNet
import secrets

import numpy as np
from facecify.fetch_frames import extract_frames_from_videos
from flask import current_app, session
from flask_mail import Mail, Message
from itsdangerous import URLSafeTimedSerializer
from werkzeug.utils import secure_filename
from flask import jsonify, render_template, redirect, url_for, request, flash
from facecify.forms import AddClassForm, AddSubjectForm, AssignSubjectForm, ClassSelectionForm, ClassSubjectSelectionForm, LoginForm, OTPVerificationForm, RegistrationForm, StudentRegistrationForm, UpdateAccountForm, VideoUploadForm
from facecify.models import Class, Student, Subject, db, Faculty
from facecify.face_recognition import train_and_save_model
from flask_login import login_user, current_user, logout_user, login_required
from facecify import app,db,bcrypt
from werkzeug.utils import secure_filename
from sklearn.preprocessing import LabelEncoder


app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = 'devmitrasharma@gmail.com'
app.config['MAIL_PASSWORD'] = 'fese mqga kcqz plpn'
mail = Mail(app)

serializer = URLSafeTimedSerializer(app.config['SECRET_KEY'])
facenet = FaceNet()

with app.app_context():
    db.create_all()

# def generate_verification_token(email):
#     return serializer.dumps(email, salt='email-confirmation-salt')

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/add_class', methods=['GET', 'POST'])
@login_required
def add_class():
    if not current_user.is_admin:
        flash('You do not have permission to access this page.', 'danger')
        return redirect(url_for('home'))

    form = AddClassForm()

    if form.validate_on_submit():
        course = form.course.data
        section = form.section.data
        branch = form.branch.data
        semester = form.semester.data
        class_code=form.class_code.data

        new_class = Class(class_code=class_code, course=course, section=section, branch=branch, semester=semester)
        db.session.add(new_class)
        db.session.commit()
        flash('Class added successfully!', 'success')
        return redirect(url_for('add_class'))
    return render_template('add_class.html', form=form)


@app.route('/add_subject', methods=['GET', 'POST'])
@login_required
def add_subject():
    if not current_user.is_admin:
        flash('You do not have permission to access this page.', 'danger')
        return redirect(url_for('home'))

    form = AddSubjectForm()

    if form.validate_on_submit():
        name = form.name.data
        subject_code = form.subject_code.data
        branch= form.branch.data

        new_subject = Subject(name=name, subject_code=subject_code,branch=branch)
        db.session.add(new_subject)
        db.session.commit()
        flash('Subject added successfully!', 'success')
        return redirect(url_for('add_subject'))
    return render_template('add_subject.html', form=form)


def save_video(video_file, enrollment_number, class_id):
    # Create the filename and path
    filename = secure_filename(f"{enrollment_number}.mp4")
    directory = os.path.join(current_app.root_path, 'static/videos', str(class_id))

    # Ensure the directory exists, create if not
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Define the file path
    file_path = os.path.join(directory, filename)

    # Save the video file to the path
    video_file.save(file_path)

    return filename

@app.route('/clear_session', methods=['POST'])
def clear_session():
    # Clear all relevant session data
    session.pop('registration_data', None)
    session.pop('otp', None)
    session.pop('otp_email', None)
    return '', 204  # No Content


@app.route('/register_student', methods=['GET', 'POST'])
def register_student():
    form = StudentRegistrationForm()
    form.class_.choices = [(c.id, c.class_code) for c in Class.query.all()]
    
    if form.validate_on_submit():
        email = form.email.data
        
        otp = str(random.randint(100000, 999999))
        
        session['otp'] = otp
        session['otp_email'] = email
        
        # Send OTP to the user's email
        subject = "Your OTP Code"
        msg = Message(subject,
                      sender='devmitrasharma@gmail.com', 
                      recipients=[email],
                      body=f"Your OTP code is {otp}")
        try:
            mail.send(msg)
            flash('OTP send successfully to your mail address!', 'success')
        except Exception as e:
            flash(f"Error sending OTP: {str(e)}", 'danger')
            return render_template('register_student.html', title='Student Registration', form=form)
        
        
        # Store registration data in session
        session['registration_data'] = {
            'name': form.name.data,
            'email': form.email.data,
            'enrollment_number': form.enrollment_number.data,
            'class_id': form.class_.data,
            'password': form.password.data
        }
        
        # Redirect to OTP verification page
        return redirect(url_for('verify_email'))
    
    return render_template('register_student.html', title='Student Registration', form=form)


@app.route('/verify_email', methods=['GET', 'POST'])
def verify_email():
    form = OTPVerificationForm()
    if form.validate_on_submit():
        otp = form.otp.data
        # Compare the entered OTP with the one stored in the session
        if otp == session.get('otp'):
            # Successful verification
            session.pop('otp', None)
            flash('Email verified successfully!', 'success')
            return redirect(url_for('add_video'))
        else:
            flash('Invalid OTP. Please try again.', 'danger')
    
    return render_template('verify_email.html', form=form)

@app.route('/add_video', methods=['GET', 'POST'])
def add_video():
    form = VideoUploadForm()
    if form.validate_on_submit():
        video_file = form.video.data
        enrollment_number = session.get('registration_data', {}).get('enrollment_number')
        class_id = session.get('registration_data', {}).get('class_id')

        # Save video and register student
        video_filename = save_video(video_file, enrollment_number,class_id)
        
        registration_data = session.get('registration_data')
        hashed_pass = bcrypt.generate_password_hash(registration_data['password']).decode('utf-8')
        student = Student(
            name=registration_data['name'],
            password=hashed_pass,
            email=registration_data['email'],
            enrollment_number=registration_data['enrollment_number'],
            video=video_filename,
            class_id=registration_data['class_id'],
            is_active=False  # Add this field to your Student model
        )

        try:
            db.session.add(student)
            db.session.commit()
            flash('Registered Successfully', 'success')
            return redirect(url_for('home'))
            # Generate verification token
            # token = generate_verification_token(student.email)
            # verify_url = url_for('verify_email', token=token, _external=True)
            # html = render_template('email_verification.html', verify_url=verify_url)
            
            # # Send verification email
            # subject = "Please confirm your email"
            # msg = Message(subject, recipients=[student.email], html=html)
            # mail.send(msg)
            
            # flash("A verification email has been sent to your email address.", 'info')
            # return redirect(url_for('login'))
        except Exception as e:
            db.session.rollback()
            flash("Error registering student: " + str(e), 'danger')
    return render_template('add_video.html', title='Add Video', form=form)

@app.route('/assign_subjects', methods=['GET', 'POST'])
@login_required
def assign_subjects():
    form = AssignSubjectForm()

    # Fetch all classes and subjects
    classes = [(c.id, f"{c.class_code} - {c.course}") for c in Class.query.all()]
    subjects = [(s.id, f"{s.name} ({s.subject_code})") for s in Subject.query.all()]

    # Initialize form choices
    form.class_.choices = [(0, 'Select a class')] + classes  # Add a 'None' option
    form.subject.choices = [(0, 'Select a subject')] + subjects  # Add a 'None' option

    if form.validate_on_submit():
        selected_class_id = form.class_.data
        selected_subject_id = form.subject.data

        # Check if 'None' was selected
        if selected_class_id == 0 or selected_subject_id == 0:
            flash('Please select both a class and a subject.', 'danger')
            return redirect(url_for('assign_subjects'))

        selected_class = Class.query.get(selected_class_id)
        selected_subject = Subject.query.get(selected_subject_id)

        if not selected_class or not selected_subject:
            flash('Invalid class or subject selected.', 'danger')
            return redirect(url_for('assign_subjects'))

        # Check if branches match
        if selected_class.branch != selected_subject.branch:
            flash('The branch of the selected class does not match the branch of the selected subject.', 'danger')
            return redirect(url_for('assign_subjects'))

        # Add the subject to the class
        selected_class.subjects.append(selected_subject)

        db.session.commit()
        flash('Subject assigned to class successfully!', 'success')
        return redirect(url_for('assign_subjects'))

    return render_template('assign_subjects.html', form=form)

@app.route('/view_students', methods=['GET', 'POST'])
@login_required
def view_students():
    form = ClassSelectionForm()
    # Filter the classes based on the currently logged-in faculty member
    form.class_.choices = [(cls.id, cls.class_code) for cls in current_user.classes]

    students = []
    selected_class = None
    if form.validate_on_submit():
        selected_class_id = form.class_.data
        students = Student.query.filter_by(class_id=selected_class_id).order_by(Student.enrollment_number).all()
        selected_class = Class.query.get(selected_class_id)

    return render_template('view_students.html', title='View Students', form=form, students=students, selected_class=selected_class)

@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html', title='DashBoard')

@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        faculty= Faculty.query.filter_by(email= form.email.data).first()
        if faculty and bcrypt.check_password_hash(faculty.password, form.password.data):
            login_user(faculty, remember= form.remember.data)
            next_page=request.args.get('next')
            flash("You've been logged in!", 'success')
            return redirect(next_page) if next_page else redirect(url_for('dashboard'))
        else:
            flash(f"Login Unsuccessful, check credentials!",'danger')
    return render_template('login.html', title='Login', form=form)

@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for('home'))

@app.route("/register", methods=['GET', 'POST'])
def register():
    form = RegistrationForm()
    if form.validate_on_submit():
        hashed_pass = bcrypt.generate_password_hash(form.password.data).decode('utf-8')
        faculty = Faculty(
            username=form.username.data,
            password=hashed_pass,
            email=form.email.data
        )
        try:
            db.session.add(faculty)
            db.session.commit()
            flash("You're registered as Faculty", 'success')
            return redirect(url_for('login'))
        except Exception as e:
            db.session.rollback()
            flash("Error registering faculty: " + str(e), 'danger')
    return render_template('register.html', title='Registration', form=form)

def save_picture(form_picture):
    random_hex = secrets.token_hex(8)
    f_name, f_ext = os.path.splitext(form_picture.filename)
    picture_fname = random_hex + f_ext
    picture_path = os.path.join(app.root_path, 'static/profile_pics', picture_fname)

    try:
        form_picture.save(picture_path)
        print(f"Saved picture to {picture_path}")  # Debugging line
    except Exception as e:
        print(f"Error saving picture: {e}")

    return picture_fname

@app.route('/account', methods=['GET', 'POST'])
@login_required
def account():
    form = UpdateAccountForm()
    if form.validate_on_submit():
        if form.picture.data:
            picture_file = save_picture(form.picture.data)
            current_user.image_file = picture_file
        else:
            print("No picture file received.")
        current_user.username = form.username.data
        current_user.email = form.email.data
        
        # Handle class selection
        selected_class_id = form.class_.data
        selected_class = Class.query.get(selected_class_id)
        if selected_class_id == 0:
            # If 'None' is selected, do nothing with the classes
            pass
        elif selected_class and selected_class not in current_user.classes:
            current_user.classes.append(selected_class)
        
        db.session.commit()
        flash("Your Account has been updated", 'success')
        return redirect(url_for('account'))
    elif request.method == 'GET':
        form.username.data = current_user.username
        form.email.data = current_user.email
        form.class_.data = 0  # Set to 'None' initially

    image_file = url_for('static', filename='profile_pics/' + current_user.image_file)
    return render_template('account.html', title='Account', image_file=image_file, form=form)

@app.route('/your_classes', methods=['GET'])
def your_classes():
    return render_template('your_classes.html', title='Classes')

@app.route('/lectures')
def lectures():
    pass

@app.route('/train_model', methods=['GET', 'POST'])
@login_required
def train_model():
    form = ClassSelectionForm()
    form.class_.choices = [(cls.id, cls.class_code) for cls in current_user.classes]

    if form.validate_on_submit():
        selected_class_id = form.class_.data
        out_directory = os.path.join(current_app.root_path, 'static', 'dataset',str(selected_class_id))
        in_directory = os.path.join(current_app.root_path, 'static', 'videos',str(selected_class_id))

        #--> if empty wala logic <--

        if not os.path.exists(out_directory):
            os.makedirs(out_directory)

        
        extract_frames_from_videos(in_directory,out_directory)
        try:
            train_and_save_model(out_directory)
            flash('Model trained', 'success')
            return redirect(url_for('home'))
        except Exception as e:
            flash(f'Error training model: {str(e)}', 'danger')
            return redirect(url_for('dashboard'))
    # if not class_id:
    #     return redirect(url_for('home'))
    # directory = os.path.join(current_app.root_path, 'static', 'dataset')
    # try:
    #         train_and_save_model(directory)
    #         flash('Model training started successfully!', 'success')
    # except Exception as e:
    #         flash(f'Error training model: {str(e)}', 'danger')
    #         return redirect(url_for('home'))
    return render_template('train_model.html', title='Training', form=form)

def load_model(class_name):
    model_path = os.path.join(current_app.root_path, 'ML_models', 'svm',f'svm_model_{class_name}.pkl')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model for class {class_name} not found")
    with open(model_path, 'rb') as model_file:
        model = pickle.load(model_file)
    return model

def load_embeddings(class_name):
    embeddings_path = os.path.join(current_app.root_path, 'ML_models', 'npz',f'faces_embeddings_{class_name}.npz')
    if not os.path.exists(embeddings_path):
        raise FileNotFoundError(f"Embeddings for class {class_name} not found")
    embeddings = np.load(embeddings_path)
    X = embeddings['arr_0']
    Y = embeddings['arr_1']
    return X, Y


@app.route('/attendance', methods=["GET", 'POST'])
@login_required
def attendance():
    form = ClassSubjectSelectionForm()
    user_classes = current_user.classes
    form.class_.choices = [(None, 'None')] + [(cls.id, cls.class_code) for cls in user_classes]
    subjects = []
    for user_class in user_classes:
        subjects.extend(user_class.subjects)
    
    form.subject.choices = [(None, 'None')] + [(subject.id, subject.name) for subject in subjects]

    if form.validate_on_submit():
        selected_class = form.class_.data
        selected_subject = form.subject.data
        selected_date = form.date.data
        
        try:
            model = load_model(selected_class)
            X, Y = load_embeddings(selected_class)
        except FileNotFoundError as e:
            return str(e)

        encoder = LabelEncoder()
        encoder.fit(Y)
        
        haarcascade_path = os.path.join(current_app.root_path, 'haarcascade_frontalface_default.xml')
        if not os.path.exists(haarcascade_path):
            flash(f'Error: Haar Cascade file not found at {haarcascade_path}', 'danger')
            return redirect(url_for('attendance'))
        
        haarcascade = cv.CascadeClassifier(haarcascade_path)
        if haarcascade.empty():
            flash('Error: Haar Cascade classifier could not be loaded', 'danger')
            return redirect(url_for('attendance'))
        
        cap = cv.VideoCapture(0)
        CONFIDENCE_THRESHOLD = 0.2
        face_frames = {}
        face_counter = {}

        facenet = FaceNet()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            rgb_img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            gray_img = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            faces = haarcascade.detectMultiScale(gray_img, 1.3, 5)

            for x, y, w, h in faces:
                img = rgb_img[y:y+h, x:x+w]
                img = cv.resize(img, (160, 160))
                img = np.expand_dims(img, axis=0)
                yhat = facenet.embeddings(img)
                face_name = model.predict(yhat)
                probabilities = model.predict_proba(yhat)
                confidence = max(probabilities[0])

                if confidence >= CONFIDENCE_THRESHOLD:
                    final_name = encoder.inverse_transform(face_name)[0]
                    face_counter[final_name] = face_counter.get(final_name, 0) + 1

                    if final_name not in face_frames:
                        face_frames[final_name] = 1
                    else:
                        face_frames[final_name] += 1

                    cv.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 255), 10)
                    cv.putText(frame, f'{final_name} ({confidence:.2f})', (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv.LINE_AA)

            cv.imshow("Face Recognition:", frame)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv.destroyAllWindows()

        # Define the filename and path for the CSV file
        date_str = selected_date.strftime('%Y-%m-%d')
        class_obj = Class.query.get(selected_class)
        subject_obj = Subject.query.get(selected_subject)
        filename = f"{date_str}_{subject_obj.name}_{class_obj.class_code}.csv"
        save_path = os.path.join(current_app.root_path, 'recognized_faces', filename)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # Save recognized faces to the CSV file
        with open(save_path, 'w', newline='') as csvfile:
            fieldnames = ['Enrollment NUmber']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for name, count in face_frames.items():
                if count > 5:
                    writer.writerow({'Enrollment NUmber': name})
        
        flash("Face recognition session ended. CSV saved ", 'success')
        return redirect(url_for('dashboard'))
    
    return render_template('attendance.html', title='Attendance', form=form)

if __name__ == "__main__":
    app.run(debug=True)
