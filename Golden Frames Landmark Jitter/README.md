Golden Frames Extraction using Landmark Jitter Sampling (README)
1) Goal (What problem we are solving)

Deepfake videos often have small, unnatural distortions around the face—especially near eyes, mouth, and nose—because synthesis and blending are imperfect.

Instead of sending all frames to heavy deepfake models, we want to select only the most “suspicious” frames.

Those selected frames are called Golden Frames.

The logic is:

If facial landmarks shift in an unnatural way from one frame to the next (after removing normal head motion), those frames are more likely to contain deepfake artifacts.

2) Pipeline overview

The script does this:

Read the video

For each frame (or sampled frames):

Detect the face region

Detect facial landmarks for key points: eyes, nose, mouth

Compare consecutive frames using a jitter score

Select frames with the highest jitter score (above a threshold)

Save:

scores.csv (all frames + jitter score + selected flag)

images of Golden Frames

optional debug images showing landmarks drawn on the face

Output structure:

output/<video_name>/
  scores.csv
  golden_frames/
  landmarks_debug/     (optional)
  frames_all/          (optional)
3) Why we detect face first, then run landmarks on a crop
Problem

If we directly run FaceMesh on the full frame, it often fails when:

face is small in the frame

face is off-center

lighting is poor

motion blur exists

Solution in our code

We do a stronger 2-step approach:

FaceDetection finds a bounding box of the face

We crop around that face (with padding), then run FaceMesh on the crop

This increases landmark detection success a lot.

4) Landmarks we use (and why)

We only use a small set of landmarks (lightweight + fast):

Eyes corners: (4 points)

Nose tip: (1 point)

Mouth corners + lip centers: (4 points)

These regions are most affected in deepfakes because:

mouth synthesis is hard during speech

eyes blink artifacts are common

nose/face boundary warping happens in blending

So we detect these points and track how they change between frames.

5) What is “Landmark Jitter”?

Landmark jitter means:

“How much landmarks move in a way that doesn’t look like normal head movement.”

Example:

In a real video, if the head moves, all landmarks shift together smoothly.

In a deepfake, sometimes only mouth landmarks jitter or eyes move weirdly even if head motion is smooth.

That “non-rigid inconsistency” is what we score.

6) Why NOT use raw Euclidean distance between frames?

A naive idea is:

score = EuclideanDistance(landmarks_t, landmarks_(t+1))

But this fails badly because Euclidean distance becomes large due to normal motion:

head turning left/right (rotation)

person moving closer/farther (scale)

camera shaking (translation)

slight viewpoint changes

So normal real video can get high Euclidean distance → false positives.

In short:

✅ Euclidean distance is sensitive to normal motion → not good.

7) What distance we use instead (best choice here)

We use:

✅ Procrustes-aligned residual distance (Similarity transform alignment + RMS error)

Meaning:

Step A: Align landmarks between frames

We take landmarks in frame A and frame B.

We compute a similarity transform that best maps A → B, allowing:

translation (shift)

rotation

scaling

This alignment removes:

camera shake

head movement

zoom

Step B: Compute residual after alignment

After alignment, we compute the leftover difference:

residual = B - aligned(A)

Then we compute RMS (root mean square) over all points:

jitter_score = RMS(residual)

Why this works

Now the score mainly captures:

warped mouth shapes

inconsistent eye geometry

subtle synthesis “wiggle”

unnatural changes that cannot be explained by head motion

So a high score means:

landmarks changed in a way that normal motion cannot explain → suspicious.

8) Thresholding (how frames are selected)

Every frame gets a jitter score (except:

first valid landmark frame

frames where face wasn’t detected)

Then we select frames using one of these modes:

A) Percentile (default)

Example: threshold_value=90

This means:

compute the 90th percentile score

select frames above it
So you keep the top 10% most suspicious frames.

This is best because it adapts to different videos.

B) Mean + Std

Select frames above:

mean + K * std

C) Fixed threshold

Select frames where score > constant.

9) Why we use min_gap

If a deepfake artifact happens in a short segment, many consecutive frames will be similar.

min_gap ensures we don’t save near-duplicate frames.

Example:
If min_gap = 3, we save a frame, then skip the next 3 sampled frames before saving another.

10) What scores.csv contains

scores.csv includes:

frame_idx: original frame number in video

jitter_score: computed RMS residual (or empty if not valid)

selected: True/False (after thresholding)

This file is important for:

plotting score curve

comparing real vs fake distribution

selecting top-K frames later if needed

11) Command usage

Install libs:

pip install -r requirements.txt

Run all frames:

python GoldenFrames.py --video sample.mp4 --out output --debug_landmarks

Run ~1 frame per second:

python GoldenFrames.py --video sample.mp4 --out output --debug_landmarks --sample_fps 1
12) Quick sanity check

After running:

landmarks_debug/ should contain images with green dots on face

scores.csv should have jitter scores

golden_frames/ should contain selected suspicious frames

If:

landmarks_debug is empty
then face detection failed → video has tiny face or extreme blur.

13) Why this module matters in the full architecture

This Golden Frames module is the “filter” stage:

Video → Golden frames → Ensemble experts (texture, frequency, etc.) → Cross-attention fusion → classifier

It reduces compute and gives the ensemble only the frames most likely to contain deepfake artifacts.