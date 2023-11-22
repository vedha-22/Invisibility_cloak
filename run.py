from flask import Flask, render_template, request, redirect, url_for, flash, Response
from flask_pymongo import PyMongo
from werkzeug.security import generate_password_hash, check_password_hash
import numpy as np
import cv2
import secrets

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)  # 16 bytes to generate a 32-character hexadecimal string

# MongoDB Configuration
app.config['MONGO_URI'] = 'mongodb://localhost:27017/UserDB'
mongo = PyMongo(app)

# User collection in MongoDB
users_collection = mongo.db.users


@app.route('/')
def index():
    # Redirect to the sign-up page
    return redirect(url_for('signup'))


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method != 'POST':
        return render_template('signup.html')

    username: str = request.form['username']
    password: str = request.form['password']

    # Check if username already exists
    if users_collection.find_one({'username': username}):
        flash('Username already exists. Choose a different one.', 'error')
        return redirect(url_for('signup'))

    # Hash the password
    hashed_password = generate_password_hash(password, method='pbkdf2:sha256')

    # Insert user into the database
    users_collection.insert_one({'username': username, 'password': hashed_password})
    flash('Registration successful! Please log in.', 'success')
    return redirect(url_for('login'))


@app.route('/login', methods=['GET', 'POST'])
def login():
    flash("hii","error")
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        # Retrieve user from the database
        user = users_collection.find_one({'username': username})

        print(f"user: {user}")  # Print the user data to the console

        # Check if the user exists and the password is correct
        if user and check_password_hash(user['password'], password):
            flash('Login successful!', 'success')
            return render_template('index.html')
        else:
            print("Login failed")  # Print a message to the console
            flash('Invalid username or password. Please try again.', 'error')

    return render_template('login.html')

def Cloak():
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    classes = []
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]

    # Load background image
    background = cv2.imread('./image.jpg')

    # Start webcam
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO object detection
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(net.getUnconnectedOutLayersNames())

        # Post-process YOLO output
        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5 and class_id == 0:  # Class ID 0 corresponds to 'person' in COCO dataset
                    # YOLO returns bounding box coordinates as a fraction of the image size
                    center_x = int(detection[0] * frame.shape[1])
                    center_y = int(detection[1] * frame.shape[0])
                    width = int(detection[2] * frame.shape[1])
                    height = int(detection[3] * frame.shape[0])

                    # Rectangle coordinates
                    x = int(center_x - width / 2)
                    y = int(center_y - height / 2)

                    boxes.append([x, y, width, height])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # Apply cloaking effect
        red_mask = np.zeros_like(frame[:, :, 0])
        for box in boxes:
            x, y, width, height = box
            red_mask[y:y + height, x:x + width] = 255

        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=2)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_DILATE, np.ones((3, 3), np.uint8), iterations=1)

        part1 = cv2.bitwise_and(background, background, mask=red_mask)
        red_free = cv2.bitwise_not(red_mask)
        part2 = cv2.bitwise_and(frame, frame, mask=red_free)
        final_output = cv2.add(part1, part2)

        # Display the final output
        cv2.imshow("Cloak", final_output)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

    return render_template('cloak.html')

@app.route('/capture_background', methods=['POST'])
def capture_background():
    #opencv for image processing
#creating a videocapture object
    cap = cv2.VideoCapture(0) #this is my webcam

    #getting the background image
    while cap.isOpened():
        ret, background = cap.read() #simply reading from the web cam
        print(ret, background)
        if ret:
            cv2.imshow("image", background)
            if cv2.waitKey(5) == ord('q'):
                #save the background image
                cv2.imwrite("image.jpg", background)
                break
    cap.release()
    cv2.destroyAllWindows()

    return render_template('cloak.html')

@app.route('/create_cloak',methods=['GET', 'POST'])
def create_cloak():
    return Response(Cloak(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)