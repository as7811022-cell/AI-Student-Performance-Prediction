import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import os

st.set_page_config(
    page_title="AI Student Performance Prediction",
    page_icon="🎓",
    layout="wide"
)

# ---------------- STUDENT DATABASE ----------------

students = {
    "aman": {
        "password": "1234",
        "name": "Aman Singh",
        "roll": "BCA001",
        "course": "Btech"
    },

    "rohit": {
        "password": "1234",
        "name": "Rohit Kumar",
        "roll": "BCA002",
        "course": "BCA"
    },

    "arnit": {
        "password": "9999",
        "name": "Arnit Chauhan",
        "roll": "BCA114",
        "course": "Btech"
    },

    "atul": {
        "password": "1010",
        "name": "Atul Singh",
        "roll": "BCA004",
        "course": "Btech"
    }
}


# ---------------- SESSION INITIALIZATION ----------------

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "student" not in st.session_state:
    st.session_state.student = None


# ---------------- LOGIN PAGE ----------------

if not st.session_state.logged_in:

    st.title("🎓 Student Performance Prediction System")

    st.subheader("Login")

    username = st.text_input("Username")
    password = st.text_input(
        "Password",
        type="password"
    )

    if st.button("Login"):

        username = username.lower().strip()

        if username in students and password == students[username]["password"]:

            st.session_state.logged_in = True
            st.session_state.student = students[username]
            st.session_state.username = username

            st.rerun()

        else:
            st.error("Invalid Username or Password")

    st.stop()



# ---------------- LOAD MODEL ----------------

try:
    model = joblib.load("model.pkl")

except:
    st.error(
        "model.pkl file not found. "
        "Put your trained model.pkl in the same folder."
    )
    st.stop()



# ---------------- STUDENT DETAILS ----------------

student = st.session_state.student

name = student["name"]
roll = student["roll"]
course = student["course"]



# ---------------- HEADER ----------------

st.title("🎓 AI Student Performance Prediction System")

st.write(
    "Machine Learning Based Student Result Prediction"
)



# ---------------- SIDEBAR ----------------

st.sidebar.title("Student Information")

st.sidebar.text_input(
    "Name",
    value=name,
    disabled=True
)

st.sidebar.text_input(
    "Roll Number",
    value=roll,
    disabled=True
)

st.sidebar.text_input(
    "Course",
    value=course,
    disabled=True
)



# ---------------- INPUT ----------------

st.header("Enter Student Details")


col1, col2 = st.columns(2)


with col1:

    study = st.slider(
        "📚 Study Hours",
        0,
        10,
        5
    )

    attendance = st.slider(
        "📅 Attendance %",
        0,
        100,
        80
    )


with col2:

    marks = st.slider(
        "📝 Previous Marks",
        0,
        100,
        70
    )

    assignment = st.slider(
        "📖 Assignment Score",
        0,
        100,
        80
    )



# ---------------- PREDICTION ----------------

if st.button("🔍 Predict Result"):


    input_data = pd.DataFrame(
        [[
            study,
            attendance,
            marks,
            assignment
        ]],
        columns=[
            "StudyHours",
            "Attendance",
            "PreviousMarks",
            "Assignments"
        ]
    )


    prediction = model.predict(input_data)


    probability = model.predict_proba(input_data)


    pass_probability = probability[0][1]*100

    fail_probability = probability[0][0]*100



    if prediction[0] == 1:

        st.success(
            "✅ Student is likely to PASS"
        )

        st.balloons()

    else:

        st.error(
            "❌ Student is likely to FAIL"
        )



    st.subheader("Prediction Confidence")

    st.write(
        f"Pass Probability: {pass_probability:.2f}%"
    )

    st.write(
        f"Fail Probability: {fail_probability:.2f}%"
    )



    score = (
        study*10 +
        attendance +
        marks +
        assignment
    )/4



    st.subheader("Performance Score")

    st.progress(
        int(score)
    )

    st.write(
        f"{score:.1f}/100"
    )



    # Chart

    st.subheader("Performance Analysis")


    fig, ax = plt.subplots()


    ax.bar(
        [
            "Study",
            "Attendance",
            "Marks",
            "Assignment"
        ],
        [
            study*10,
            attendance,
            marks,
            assignment
        ]
    )


    ax.set_ylim(0,100)


    st.pyplot(fig)



    # Save history

    history = pd.DataFrame([{

        "Name":name,
        "Roll":roll,
        "Course":course,
        "Study Hours":study,
        "Attendance":attendance,
        "Marks":marks,
        "Assignment":assignment,
        "Result":
        "PASS" if prediction[0]==1 else "FAIL"

    }])


    if os.path.exists("history.csv"):

        history.to_csv(
            "history.csv",
            mode="a",
            header=False,
            index=False
        )

    else:

        history.to_csv(
            "history.csv",
            index=False
        )


    st.success(
        "Prediction saved"
    )



# ---------------- HISTORY ----------------

if os.path.exists("history.csv"):

    st.subheader("📜 Prediction History")

    history = pd.read_csv(
        "history.csv"
    )

    st.dataframe(history)



# ---------------- LOGOUT ----------------

if st.sidebar.button("Logout"):

    st.session_state.logged_in = False
    st.session_state.student = None

    st.rerun()




