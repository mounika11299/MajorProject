from pathlib import Path
# Imports for streamlit
import streamlit as st
import av
import cv2
from streamlit_webrtc import (
    RTCConfiguration,
    VideoProcessorBase,
    WebRtcMode,
    webrtc_streamer,
)
# Model Loading imports
import numpy as np
import mediapipe as mp
from keras.models import load_model
import streamlit as st
import webbrowser
import streamlit.components.v1 as components
from streamlit_player import st_player

# Fetch tracks from spotify using spotify api (future use)
# import spotipy
# from spotipy.oauth2 import SpotifyClientCredentials

# # Set up authentication
# client_credentials_manager = SpotifyClientCredentials(
#     client_id='90376e83503746b695b31e1ab130f596', client_secret='36d18f7b1898428996805cb39443d74a')
# sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

# # Search for playlists
# results = sp.search(q='mood:calm', type='playlist', limit=10)

# # Retrieve tracks from the first playlist found
# if results['playlists']['items']:
#     playlist_id = results['playlists']['items'][0]['id']
#     playlist_name = results['playlists']['items'][0]['name']
#     playlist_tracks = sp.playlist_tracks(playlist_id)

#     # Display playlist name and tracks
#     print(f"Playlist: {playlist_name}")
#     for track in playlist_tracks['items']:
#         print(f"- {track['track']['name']} by {track['track']['artists'][0]['name']}")
# else:
#     print("No playlists found for the specified mood.")


st.set_page_config(
    page_title="Emotion Based Music Recommendation 🎵",
    page_icon="🎵",
)

page_bg_img = """
<style>

div.stButton > button {
    width: 220px;
    height: 50px;
    font-size: 36px;
    color: white;
    z-index: 1;
    padding: 10px 15px;
    display: flex;
    align-items: center;
    justify-content: center;
    white-space: nowrap;
    border-radius: 10px;
    background: #0f172a;
    box-shadow: 0 0 1px 1px white;
}

div.stButton > button:hover {
    transform: scale(1.05);
    transition: transform 0.5s ease-in;
    box-shadow: none;
}

[data-testid="stAppViewContainer"] {
    background-image: url("https://images.unsplash.com/photo-1613327986042-63d4425a1a5d?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1470&q=80");
    background-size: cover;
    background-position: top left;
    background-repeat: no-repeat;
    background-attachment: fixed;
}

[data-testid="stSidebar"] > div:first-child {
    background-position: center;
    background-repeat: no-repeat;
    background-attachment: fixed;
    background : #0f172a;
}

[data-testid="stSidebarUserContent"] {
    max-height: 100vh;
    content: fit;
    padding: 4rem 1rem;
}

[data-testid="stHeader"] {
    background: transparent;
    color: white
}

[data-testid="stToolbar"] {
    right: 2rem;
    background : #0f001a;
}

[data-testid="stAppViewBlockContainer"]{
    padding: 1.25rem;
}

div.stMarkdown ,h1{
    color:#f5f5f5;
    background:transparent;
}

[data-testid="StyledLinkIconContainer"]{
    font:20px;
    color:white;
    align-items:center;
    background : transparent;
}

[data-testid ="stText"]{
    font-size:18px;
    align-items:center;
    color:#177233;
    background:#e8f9ee;
    width:100%;
    padding:10px 15px;
    border-radius:10px;
}

[data-testid ="stWidgetLabel"]{
    color:white;
}

</style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)
st.title("Emotion Based Music Recommendation 🎉")
st.sidebar.image("sidebar.png")
st.sidebar.success("Emotion Based Music Recommendation 🎉")

st.markdown("**Hey there, emotion explorer! Are you ready for a wild ride through the rollercoaster of feelings?**")
st.markdown("**Welcome to EMR, where our snazzy AI meets your wacky emotional world head-on! We've got our virtual goggles on (nope, not really, but it sounds cool** 😎 **) to analyze your emotions using a webcam. And what do we do with all those emotions, you ask? We turn them into the most toe-tapping, heartwarming, and occasionally hilarious music playlists you've ever heard!** 🕺💃")
st.markdown("**So, strap in** 🚀 **, hit that webcam** 📷 **, and let the musical journey begin! Vibescape is your ticket to a rollercoaster of emotions, all set to your favorite tunes.** 🎢")

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{
        "urls": ["stun:stun.l.google.com:19302"]
    }]})

# CWD path
HERE = Path(__file__).parent

model = load_model("models/model.h5", compile=False)
label = np.load("models/label.npy", allow_pickle=True)

holistic = mp.solutions.holistic
hands = mp.solutions.hands
holis = holistic.Holistic()
drawing = mp.solutions.drawing_utils

# Initialize session state
if "run" not in st.session_state:
    st.session_state["run"] = ""

# if "lang" not in st.session_state:
#     st.session_state["lang"] = ""

# if "emotion" not in st.session_state:
#     st.session_state["emotion"] = ""

run = np.load("models/emotion.npy", allow_pickle=True)[0]
emotion = np.load("models/emotion.npy", allow_pickle=True)[0]
#if emotion.size>0:
    #emotion=emotion[0]
lang = np.load("models/lang.npy", allow_pickle=True)[0]

try:
    emotion = np.load("models/emotion.npy", allow_pickle=True)[0]
    #if emotion.size>0:
        #emotion=emotion[0]
    lang = np.load("models/lang.npy", allow_pickle=True)[0]
    print("old emotion is ", emotion)
    print("old language is ", lang)

except:
    emotion = ""
    lang = ""


class EmotionProcessor(VideoProcessorBase):
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        frm = frame.to_ndarray(format="bgr24")
        frm = cv2.flip(frm, 1)
        res = holis.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))
        lst = []

        if res.face_landmarks:
            for i in res.face_landmarks.landmark:
                lst.append(i.x - res.face_landmarks.landmark[1].x)
                lst.append(i.y - res.face_landmarks.landmark[1].y)

            if res.left_hand_landmarks:
                for i in res.left_hand_landmarks.landmark:
                    lst.append(i.x - res.left_hand_landmarks.landmark[8].x)
                    lst.append(i.y - res.left_hand_landmarks.landmark[8].y)
            else:
                for i in range(42):
                    lst.append(0.0)

            if res.right_hand_landmarks:
                for i in res.right_hand_landmarks.landmark:
                    lst.append(i.x - res.right_hand_landmarks.landmark[8].x)
                    lst.append(i.y - res.right_hand_landmarks.landmark[8].y)
            else:
                for i in range(42):
                    lst.append(0.0)

            lst = np.array(lst).reshape(1, -1)

            pred = label[np.argmax(model.predict(lst))]
            print(pred)
            cv2.putText(frm, pred, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            np.save("models/emotion.npy", np.array([pred]))
            emotion = pred
            # st.session_state["emotion"] = pred

        drawing.draw_landmarks(frm, res.face_landmarks, holistic.FACEMESH_TESSELATION,
                               landmark_drawing_spec=drawing.DrawingSpec(
                                   color=(0, 0, 255), thickness=-1, circle_radius=1),
                               connection_drawing_spec=drawing.DrawingSpec(thickness=1))
        drawing.draw_landmarks(frm, res.left_hand_landmarks, hands.HAND_CONNECTIONS)
        drawing.draw_landmarks(frm, res.right_hand_landmarks, hands.HAND_CONNECTIONS)

        return av.VideoFrame.from_ndarray(frm, format="bgr24")


if st.session_state["run"] == "true":
    webrtc_streamer(key="key", desired_playing_state=st.session_state.get("run", "") == "true", mode=WebRtcMode.SENDRECV,  rtc_configuration=RTC_CONFIGURATION, video_processor_factory=EmotionProcessor, media_stream_constraints={
        "video": True,
        "audio": False
    }, async_processing=True)

if emotion:
    language = st.sidebar.selectbox("Select Language", ["Telugu", "Hindi", "English"])
    if language !="":
        lang = language
        st.session_state["lang"] = lang
        

col1, col2, col6 = st.columns([1, 1, 1])

with col1:
    if emotion :
        text =  "Recapture" 
    else :
        text = "Capture"
    start_btn = st.button(text)

with col6:
    if st.session_state["run"] == "true":
        stop_btn = st.button("Stop")

if start_btn:
    st.session_state["run"] = "true"
    st.rerun()

if st.session_state["run"] == "true" and stop_btn:
    st.session_state["run"] = "false"
    st.rerun()
else:
    if not emotion:
        pass
    else:
        # np.save("models/emotion.npy", np.array([""]))
        st.session_state["emotion"] = run
        st.text("Your current emotion is " + emotion)
        st.subheader("Choose your streaming service")

col3, col6, col10 = st.columns([1, 1, 1])

# Spotify playlists
spotify_hindi_playlist_urls = {
    "Happy": "https://open.spotify.com/embed/playlist/37i9dQZF1DWTwbZHrJRIgD?utm_source=generator&theme=0",
    "Sad": "https://open.spotify.com/embed/playlist/37i9dQZF1DXdFesNN9TzXT?utm_source=generator&theme=0",
    "Angry": "https://open.spotify.com/embed/playlist/2r9tRVoHG3AMBTvKJ8abOl?utm_source=generator&theme=0",
    "Fear": "https://open.spotify.com/embed/playlist/7EZ4lWeM1OLxZYfGmhDbJI?utm_source=generator&theme=0",
    "Surprise": "https://open.spotify.com/embed/playlist/7vatYrf39uVaZ8G2cVtEik?utm_source=generator&theme=0",
    "Neutral": "https://open.spotify.com/embed/playlist/37i9dQZF1DX0XUfTFmNBRM?utm_source=generator&theme=0"
}

spotify_english_playlist_urls = {
    "Happy": "https://open.spotify.com/embed/playlist/37i9dQZF1DXdPec7aLTmlC?utm_source=generator&theme=0",
    "Sad": "https://open.spotify.com/embed/playlist/37i9dQZF1DX7qK8ma5wgG1?utm_source=generator&theme=0",
    "Angry": "https://open.spotify.com/embed/playlist/37i9dQZF1EIgNZCaOGb0Mi?utm_source=generator&theme=0",
    "Fear": "https://open.spotify.com/embed/playlist/37i9dQZF1EIfgYPpPEriFK?utm_source=generator&theme=0",
    "Surprise": "https://open.spotify.com/embed/playlist/37i9dQZF1EIfIhwClkyzKs?utm_source=generator&theme=0",
    "Neutral": "https://open.spotify.com/embed/playlist/37i9dQZEVXbMDoHDwVN2tF?utm_source=generator&theme=0",
    "Default": "https://open.spotify.com/embed/playlist/60eEo5VdhekyGJKTjK2xSV?utm_source=generator&theme=0"
}

spotify_telugu_playlist_urls = {
    "Happy": "https://open.spotify.com/embed/playlist/37i9dQZF1DX2UT3NuRgcHd?utm_source=generator&theme=0",
    "Sad": "https://open.spotify.com/embed/playlist/37i9dQZF1DWUEWjDsV7AgX?utm_source=generator&theme=0",
    "Angry": "https://open.spotify.com/embed/playlist/37i9dQZF1DX4H5837Y8I1n?utm_source=generator&theme=0",
    "Fear": "https://open.spotify.com/embed/playlist/4JVDfzNmjX17JnfjaziVQe?utm_source=generator&theme=0",
    "Surprise": "https://open.spotify.com/embed/playlist/37i9dQZF1DXcuAVsi45c0d?utm_source=generator&theme=0",
    "Neutral": "https://open.spotify.com/embed/playlist/37i9dQZF1DX6XE7HRLM75P?utm_source=generator&theme=0"
}

# Spotify Player
with col3:
    if emotion:
        btn1 = st.button("Spotify")
        if btn1:
            st.sidebar.success("Spotify has been selected as your music player.")
            if "run" not in st.session_state:
                st.write("**Looks like you have skipped the face scan on the homepage and came here, just for music, just choose your vibe manually for Vibescape to groove with you!**")
                option = st.selectbox(
                    'What''s your vibe today?',
                    ('Happy', 'Sad', 'Angry', 'Fear', 'Surprise', 'Neutral'))
                st.session_state["emotion"] = option
            else:
                emotion = st.session_state["emotion"]

if emotion and lang and btn1:
    print("Spotify Chosen")
    print("Emotion is ", emotion)
    print("Language is ", lang)
    st.header("Spotifiy Music Recommendation 💚")
    if language == "Hindi":
        st.subheader('Hindi Music')
        playlist_url = spotify_hindi_playlist_urls.get(emotion)
        components.iframe(playlist_url, height=610, scrolling=True)

    elif language == "Telugu":
        st.subheader("Telugu Music")
        playlist_url = spotify_telugu_playlist_urls.get(emotion)
        components.iframe(playlist_url, height=610, scrolling=True)

    elif language == "English":
        st.subheader('English Music')
        playlist_url = spotify_english_playlist_urls.get(emotion)
        components.iframe(playlist_url, height=610, scrolling=True)


# SoundCloud Playlists
soundcloud_hindi_playlist_urls = {
    "Happy": "https://soundcloud.com/miss_happy/sets/hindi-songs",
    "Sad": "https://soundcloud.com/aryan-ambuj-752291555/sets/sad-hindi-songs-a-little",
    "Angry": "https://soundcloud.com/user-905375441/sets/hindi-rap-songs",
    "Fear": "https://soundcloud.com/narendraswapnil/sets/aavirbhaav-a-hindi-horror",
    "Surprise": "https://soundcloud.com/maryam-zeb-592867892/sets/hot-hindi-hits-2010-2020",
    "Neutral": "https://soundcloud.com/user635881277/sets/hindi-hits",
}

soundcloud_telugu_playlist_urls = {
    "Happy": "https://soundcloud.com/sumit-indoria/sets/telugu-party-time",
    "Sad": "https://soundcloud.com/user-738522704/sets/sad-telugu-songs",
    "Angry": "https://soundcloud.com/user-692822299/sets/telugu-workout-remix",
    "Fear": "https://soundcloud.com/vinod-kumar-761560211/sets/telugu-songs-regular-update",
    "Surprise": "https://soundcloud.com/vinod-kumar-761560211/sets/telugu-songs-regular-update",
    "Neutral": "https://soundcloud.com/vinod-kumar-761560211/sets/telugu-songs-regular-update",
}

soundcloud_english_playlist_urls = {
    "Happy": "https://soundcloud.com/gabriela-astudillo-398435247/sets/happy-english-music",
    "Sad": "https://soundcloud.com/jishnu-rajwani-695997535/sets/famous-english-sad-songs-of",
    "Angry": "https://soundcloud.com/thomashayden/sets/tech-house-vibes-only",
    "Fear": "https://soundcloud.com/tito-tito-675324717/sets/horror-english",
    "Surprise": "https://soundcloud.com/manea-claudia/sets/top-love-songs-2022-playlist-1",
    "Neutral": "https://soundcloud.com/sejal-agarkar/sets/english-songs-hits",
}

# Soundcloud Player
with col6:
    if emotion:
        btn2 = st.button("Soundcloud")
        if btn2:
            st.sidebar.success("SoundCloud has been selected as your music player.")
            if "emotion" not in st.session_state:
                st.write("**Looks like you have skipped the face scan on the homepage and came here, just for music, just choose your vibe manually for Vibescape to groove with you!**")
                option = st.selectbox(
                    'What''s your vibe today?',
                    ('Happy', 'Sad', 'Angry', 'Fear', 'Surprise', 'Neutral'))
                st.session_state["emotion"] = option
            else:
                emotion = st.session_state["emotion"]

if emotion and lang and btn2:
    print("SoundCloud Chosen")
    print("Emotion is ", emotion)
    print("Language is ", lang)
    st.header("SoundCloud Music Recommendation ❤️")
    if language == "Hindi":
        st.subheader('Hindi Music')
        playlist_url = soundcloud_hindi_playlist_urls.get(emotion)
        st_player(playlist_url, height=610)

    elif language == "Telugu":
        st.subheader("Telugu Music")
        playlist_url = soundcloud_telugu_playlist_urls.get(emotion)
        st_player(playlist_url, height=610)

    elif language == "English":
        st.subheader('English Music')
        playlist_url = soundcloud_english_playlist_urls.get(emotion)
        st_player(playlist_url, height=610)


# Youtube Player
with col10:
    if emotion:
        btn3 = st.button("Youtube")
        st.sidebar.warning("For Youtube, you will be redirected")
        if lang and btn3:
            st.sidebar.success("YouTube has been selected as your music player.")
            if "run" not in st.session_state:
                st.write("**Looks like you have skipped the face scan on the homepage and came here, just for music, just choose your vibe manually for Vibescape to groove with you!**")
                option = st.selectbox(
                    'What''s your vibe today?',
                    ('Happy', 'Sad', 'Angry', 'Fear', 'Surprise', 'Neutral'))
                st.session_state["emotion"] = option
            else:
                emotion = st.session_state["emotion"]
                print("Youtube Chosen")
                print("Emotion is ",emotion)
                print("Language is ", lang)
                webbrowser.open(f"https://www.youtube.com/results?search_query={lang}+{emotion}+songs")
