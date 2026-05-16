# MoodSense

I worked on MoodSense as a machine learning project that does real-time emotion detection through your webcam. The idea was to go beyond just detecting a face and actually track behavioral patterns over time, things like how stable your mood is, how long you stay in one emotion, and whether stress signals are building up.

---

## What it does

When you run it, a browser window opens automatically and you just click Start Camera. From that point it is reading your facial expressions live and showing you a full analytics dashboard in real time.

Here is what you get:

**Live camera feed** with emotion-colored bounding boxes drawn over every face it detects. Each box has corner bracket decorations and a label showing the emotion and confidence percentage.

**Dominant emotion card** on the right side that updates as your expression changes. It shows a large emotion name, an emoji, the confidence score, and a colored bar that fills up based on how confident the model is.

**All emotions panel** showing all seven emotions as live updating bars so you can see the full breakdown of what the model is picking up, not just the top one.

**Streak tracker** that counts how many consecutive frames have had the same dominant emotion. So if you have been neutral for 40 frames it tells you that.

**Emotion timeline chart** plotting the last two minutes of emotion confidence as a line graph with one trace per emotion, so you can see patterns over time.

**Session distribution chart** showing what percentage of your total session each emotion has dominated.

**Session stats row** with five cards: total session time, your dominant mood for the whole session, a stress level indicator (low, medium, high), a mood stability score based on how often your emotion switches, and your longest streak of the session.

**Stress and mood alerts** that pop up as a banner when stress signals stay elevated or when you hit a long happy streak.

**CSV export** so you can download your full session data with timestamps, emotion labels, and confidence scores.

---

## How to run it

Install the dependencies:

```
py -3.11 -m pip install -r requirements.txt
```

Then just run:

```
py -3.11 moodsense.py
```

The browser opens automatically. Click Start Camera and allow webcam access. There will be a short loading spinner the first time while the model warms up, then it starts detecting right away.

---

## Settings

There is a Settings drop down in the top right with a few options. The most useful one is the analysis interval slider which controls how often a frame gets sent for analysis. I set the default to 200ms which gives you smooth detection. If your machine is struggling you can push it to 400 or 500ms.

You can also toggle each panel on and off individually and turn off the scan line effect on the camera if you find it distracting.

---

## Stack

- Flask for the backend server
- DeepFace with OpenCV backend for the emotion analysis
- OpenCV for decoding the frames the browser sends over
- Chart.js for the live charts
- Vanilla HTML, CSS, and JavaScript for the frontend, no framework needed

---

## Notes

The video and the analysis are completely separate. The camera feed runs at your webcam's native frame rate through the browser's own video element, and the analysis just sends small snapshots in the background every 200ms. So the video is always smooth regardless of how long DeepFace takes on any given frame.

All the TensorFlow noise is suppressed so the terminal only shows the two lines it actually needs to show.
