import json
import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import time
from collections import deque, Counter

# Set page configuration
st.set_page_config(
    page_title="Emotion-Based Storytelling",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Make sure we have a place to store the current emotion that persists across reruns
if "current_emotion" not in st.session_state:
    st.session_state.current_emotion = "neutral"

# Global variables
story_continuation = ""
emotion_update_interval = 5

# Load emotion recognition model
@st.cache_resource
def load_emotion_model():
    try:
        model = load_model('best_fer_model.h5')
        return model
    except Exception as e:
        st.error(f"Error loading emotion model: {e}")
        return None

# Load BART model for story generation
@st.cache_resource
def load_bart_model():
    try:
        model_path = "final_model"
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading BART model: {e}")
        return None, None

# Load base stories
@st.cache_data
def load_base_stories():
    try:
        with open("base_stories.json", "r") as f:
            stories = json.load(f)
        return stories
    except Exception as e:
        st.error(f"Error loading base stories: {e}")
        return []

# Create a callback to update the emotion
def update_emotion(emotion_text):
    st.session_state.current_emotion = emotion_text

class EmotionVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.emotion_model = load_emotion_model()
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
        self.frame_count = 0
        # For debugging
        if "emotion_queue" not in st.session_state:
            st.session_state.emotion_queue = []
        if "emotion_stats" not in st.session_state:
            st.session_state.emotion_stats = Counter()

    def recv(self, frame):
        self.frame_count += 1
        # Process every frame for better responsiveness
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            roi_gray = gray[y:y+h, x:x+w]

            if roi_gray.size > 0:
                roi = cv2.resize(roi_gray, (48, 48))
                roi = roi / 255.0
                roi = np.expand_dims(roi, axis=0)
                roi = np.expand_dims(roi, axis=-1)

                prediction = self.emotion_model.predict(roi, verbose=0)
                emotion_idx = np.argmax(prediction)
                emotion_text = self.emotions[emotion_idx]
                confidence = float(prediction[0][emotion_idx])

                # Update the current emotion directly
                update_emotion(emotion_text)
                
                # For debugging
                if len(st.session_state.emotion_queue) >= 30:
                    st.session_state.emotion_queue.pop(0)
                st.session_state.emotion_queue.append(emotion_text)
                st.session_state.emotion_stats[emotion_text] += 1

                cv2.putText(img, f"{emotion_text} ({confidence:.2f})",
                            (x, y-10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.9, (36, 255, 12), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

def get_current_emotion():
    return st.session_state.current_emotion

def generate_story_continuation(model, tokenizer, base_context, emotion, summary_so_far):
    try:
        input_text = f"emotion: {emotion}\ncontext: {base_context}\nsummary_so_far: {summary_so_far}"

        inputs = tokenizer(input_text, return_tensors="pt", max_length=1024, truncation=True)

        outputs = model.generate(
            inputs.input_ids,
            max_length=120,
            min_length=30,
            num_beams=2,
            temperature=0.7,
            top_k=40,
            top_p=0.9,
            no_repeat_ngram_size=2,
            early_stopping=True,
            do_sample=True
        )

        continuation = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return continuation

    except Exception as e:
        st.error(f"Error generating story: {e}")
        return "The story continues, but something feels off..."

def main():
    global emotion_update_interval, story_continuation

    st.title("Emotion-Based Interactive Storytelling")

    if "story_text" not in st.session_state:
        st.session_state.story_text = ""
    if "summary_so_far" not in st.session_state:
        st.session_state.summary_so_far = ""

    st.sidebar.title("Settings")
    emotion_update_interval = st.sidebar.slider(
        "Emotion Detection Interval (seconds)", 2, 20, 5
    )

    # Display debug info in sidebar if requested
    if st.sidebar.checkbox("Debug Emotion Detection"):
        st.sidebar.write("Current emotion queue:")
        st.sidebar.write(st.session_state.emotion_queue)
        st.sidebar.write("Emotion stats:")
        st.sidebar.write(dict(st.session_state.emotion_stats))

    emotion_model = load_emotion_model()
    bart_model, tokenizer = load_bart_model()
    stories = load_base_stories()

    if not emotion_model or not bart_model or not stories:
        st.error("Failed to load required components")
        return

    st.header("Select a Story to Begin")
    cols = st.columns(3)

    for i, story in enumerate(stories):
        with cols[i % 3]:
            st.subheader(story["title"])
            if "image" in story and os.path.exists(story["image"]):
                st.image(story["image"], caption=story["title"])
            else:
                st.markdown("📚")

            st.write(f"**Genre:** {story['genre']}")
            st.write(story["summary"])

            if st.button(f"Select '{story['title']}'", key=f"select_{i}"):
                st.session_state.selected_story = story
                st.session_state.story_started = True
                st.session_state.story_text = ""
                st.session_state.summary_so_far = story["full_context"]
                st.rerun()

    if "story_started" in st.session_state and st.session_state.story_started:
        selected_story = st.session_state.selected_story
        col1, col2 = st.columns([1, 2])

        with col1:
            st.header("Emotion Detection")
            webrtc_ctx = webrtc_streamer(
                key="emotion-detector",
                video_processor_factory=EmotionVideoProcessor,
                media_stream_constraints={"video": True, "audio": False},
                async_processing=True,
            )
            
            # Create a placeholder for real-time emotion display
            emotion_display = st.empty()
            
            # Show current emotion - this gets refreshed with the button
            current_emotion = get_current_emotion()
            
            
           

        with col2:
            st.header(f"Story: {selected_story['title']}")
            base_context = selected_story["full_context"]
            st.markdown(f"**Beginning:**\n\n{base_context}")
            st.markdown("### Story Continues...")
            story_container = st.empty()

            story_container.markdown(st.session_state.story_text or "Click 'Generate Next Part' to continue the story.")

            if st.button("Generate Next Part"):
                # Get the current emotion at the moment the button is clicked
                emotion_to_use = get_current_emotion()
                continuation = generate_story_continuation(
                    bart_model, tokenizer,
                    base_context,
                    emotion_to_use,
                    st.session_state.summary_so_far
                )
                st.session_state.story_text += "\n\n" + continuation
                st.session_state.summary_so_far += " " + continuation
                st.rerun()

            if st.button("Choose Another Story"):
                # Keep the current emotion but reset other story-related state
                current_emotion = st.session_state.current_emotion
                st.session_state.clear()
                st.session_state.current_emotion = current_emotion
                st.rerun()

if __name__ == "__main__":
    main()