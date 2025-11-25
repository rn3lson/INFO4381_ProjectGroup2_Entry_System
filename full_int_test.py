# USAGE
# python detect_mask_temp_video_headless.py

# import the necessary packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array 
from tensorflow.keras.models import load_model
from picamera2 import Picamera2
import board
import adafruit_mlx90614
import numpy as np
import argparse
import imutils
import time
import cv2
import os
from datetime import datetime

# Temperature thresholds (in Celsius) - optimized for WRIST measurement
TEMP_MIN = 32.0  # Minimum valid temperature (to verify sensor is reading a person, not air)
TEMP_MAX = 37.0  # Fever threshold for wrist (98.6°F) - wrist runs cooler than forehead

# State machine states
STATE_WAITING_FOR_FACE = "WAITING_FOR_FACE"
STATE_FACE_DETECTED = "FACE_DETECTED"
STATE_WAITING_FOR_WRIST = "WAITING_FOR_WRIST"
STATE_COMPLETE = "COMPLETE"

def detect_and_predict_mask(frame, faceNet, maskNet):
	# grab the dimensions of the frame and then construct a blob
	# from it
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
		(104.0, 177.0, 123.0))

	# pass the blob through the network and obtain the face detections
	faceNet.setInput(blob)
	detections = faceNet.forward()

	# initialize our list of faces, their corresponding locations,
	# and the list of predictions from our face mask network
	faces = []
	locs = []
	preds = []

	# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with
		# the detection
		confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the confidence is
		# greater than the minimum confidence
		if confidence > args["confidence"]:
			# compute the (x, y)-coordinates of the bounding box for
			# the object
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# ensure the bounding boxes fall within the dimensions of
			# the frame
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			# extract the face ROI, convert it from BGR to RGB channel
			# ordering, resize it to 224x224, and preprocess it
			face = frame[startY:endY, startX:endX]
			if face.any():
				face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
				face = cv2.resize(face, (224, 224))
				face = img_to_array(face)
				face = preprocess_input(face)

				# add the face and bounding boxes to their respective
				# lists
				faces.append(face)
				locs.append((startX, startY, endX, endY))

	# only make a predictions if at least one face was detected
	if len(faces) > 0:
		# for faster inference we'll make batch predictions on *all*
		# faces at the same time rather than one-by-one predictions
		# in the above `for` loop
		faces = np.array(faces, dtype="float32")
		preds = maskNet.predict(faces, batch_size=32)

	# return a 2-tuple of the face locations and their corresponding
	# locations
	return (locs, preds)

def check_pass_fail(has_mask, temperature):
	"""
	Determine if person passes screening (WRIST temperature measurement)
	"""
	if temperature < TEMP_MIN:
		return None, None, "Invalid temperature reading"
	
	has_fever = temperature > TEMP_MAX
	
	if has_mask and not has_fever:
		return "PASS", (0, 255, 0), f"Wrist temp: {temperature:.1f}C - Normal"  # Green
	elif not has_mask:
		return "FAIL", (0, 0, 255), "No mask detected"  # Red
	else:  # has fever
		return "FAIL", (0, 0, 255), f"Wrist temp: {temperature:.1f}C - FEVER DETECTED"  # Red

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--face", type=str,
	default="face_detector",
	help="path to face detector model directory")
ap.add_argument("-m", "--model", type=str,
	default="mask_detector.h5",
	help="path to trained face mask detector model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
ap.add_argument("-s", "--save-images", action="store_true",
	help="save detection images to disk")
ap.add_argument("-t", "--temp-max", type=float, default=37.0,
	help="fever threshold in Celsius for wrist (default: 37.0C / 98.6F)")
ap.add_argument("--temp-min", type=float, default=32.0,
	help="minimum valid temperature in Celsius (default: 32.0C / 89.6F)")
ap.add_argument("--wrist-timeout", type=float, default=10.0,
	help="seconds to wait for wrist after face detection (default: 10)")
ap.add_argument("--cooldown", type=float, default=5.0,
	help="seconds to wait between successful screenings (default: 5)")
args = vars(ap.parse_args())

# Update temperature thresholds if provided
TEMP_MAX = args["temp_max"]
TEMP_MIN = args["temp_min"]
WRIST_TIMEOUT = args["wrist_timeout"]
COOLDOWN_PERIOD = args["cooldown"]

# Create output directory for saved images if needed
if args["save_images"]:
	output_dir = "detections"
	os.makedirs(output_dir, exist_ok=True)

# load our serialized face detector model from disk
print("[INFO] loading face detector model...")
prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
weightsPath = os.path.sep.join([args["face"],
	"res10_300x300_ssd_iter_140000.caffemodel"])
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# load the face mask detector model from disk
print("[INFO] loading face mask detector model...")
maskNet = load_model(args["model"], compile=False)

# Initialize temperature sensor
print("[INFO] initializing temperature sensor...")
try:
	i2c = board.I2C()
	mlx = adafruit_mlx90614.MLX90614(i2c)
	print("[INFO] Temperature sensor initialized successfully")
except Exception as e:
	print(f"[ERROR] Failed to initialize temperature sensor: {e}")
	print("[ERROR] Exiting...")
	exit(1)

# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (640, 480), "format": "RGB888"})
picam2.configure(config)
picam2.start()
time.sleep(2.0)

print("[INFO] Running 2-stage mask + wrist temperature screening...")
print("[INFO] STAGE 1: Show your face to camera for mask detection")
print("[INFO] STAGE 2: Place wrist near sensor for temperature check")
print(f"[INFO] Valid wrist temperature range: {TEMP_MIN}°C - {TEMP_MAX}°C ({TEMP_MIN * 9/5 + 32:.1f}°F - {TEMP_MAX * 9/5 + 32:.1f}°F)")
print(f"[INFO] Wrist timeout: {WRIST_TIMEOUT} seconds")
print(f"[INFO] Cooldown between successful checks: {COOLDOWN_PERIOD} seconds")
print("[INFO] Press Ctrl+C to quit")
if args["save_images"]:
	print(f"[INFO] Saving detection images to '{output_dir}/' directory")

# State machine variables
current_state = STATE_WAITING_FOR_FACE
saved_mask_status = None
saved_mask_confidence = None
wrist_check_start_time = None
frame_count = 0
baseline_temp = None  # Store baseline temperature to detect wrist presence
temp_readings = []  # Store recent readings for stability check

try:
	# loop over the frames from the video stream
	while True:
		# grab the frame from the threaded video stream
		frame = picam2.capture_array()
		frame_rgb = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
		frame_resized = imutils.resize(frame_rgb, width=400)

		# Read temperature from sensor
		try:
			object_temp = mlx.object_temperature
			ambient_temp = mlx.ambient_temperature
		except Exception as e:
			object_temp = 0.0
			ambient_temp = 0.0

		# STATE MACHINE LOGIC
		if current_state == STATE_WAITING_FOR_FACE:
			# Looking for face and mask detection
			(locs, preds) = detect_and_predict_mask(frame_resized, faceNet, maskNet)
			
			if len(locs) > 0:
				# Face detected! Check mask status
				(box, pred) = (locs[0], preds[0])
				(mask, withoutMask) = pred
				
				has_mask = mask > withoutMask
				mask_confidence = max(mask, withoutMask) * 100
				
				# Save the mask status
				saved_mask_status = has_mask
				saved_mask_confidence = mask_confidence
				
				mask_label = "Mask" if has_mask else "No Mask"
				
				timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
				print(f"\n{'='*60}")
				print(f"[{timestamp}]")
				print(f"  STAGE 1: {mask_label} detected ({mask_confidence:.2f}%)")
				
				if has_mask:
					# Mask detected - proceed to temperature check
					print(f"  >>> Now place your WRIST near the sensor <<<")
					print(f"{'='*60}")
					current_state = STATE_WAITING_FOR_WRIST
					wrist_check_start_time = time.time()
				else:
					# No mask - immediate FAIL
					print(f"  >>> RESULT: FAIL <<<")
					print(f"  Message: No mask detected - Temperature check skipped")
					print(f"{'='*60}")
					
					# Save failure image if enabled
					if args["save_images"]:
						info_frame = frame_rgb.copy()
						result_color = (0, 0, 255)  # Red
						
						cv2.putText(info_frame, f"MASK: NO MASK ({mask_confidence:.1f}%)", 
							(10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, result_color, 2)
						cv2.putText(info_frame, f"WRIST TEMP: SKIPPED", 
							(10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, result_color, 2)
						cv2.putText(info_frame, f"RESULT: FAIL", 
							(10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, result_color, 3)
						cv2.putText(info_frame, "No mask detected", 
							(10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, result_color, 2)
						
						filename = f"{output_dir}/FAIL_{frame_count:05d}.jpg"
						cv2.imwrite(filename, info_frame)
						print(f"[INFO] Saved detection image: {filename}")
						frame_count += 1
					
					# Move to complete state
					current_state = STATE_COMPLETE
			else:
				# No face detected yet
				if time.time() % 5 < 0.5:  # Print every 5 seconds
					print("[INFO] Waiting for person... Please show your face to the camera")
		
		elif current_state == STATE_WAITING_FOR_WRIST:
			# Check for timeout
			elapsed = time.time() - wrist_check_start_time
			if elapsed > WRIST_TIMEOUT:
				print(f"\n[TIMEOUT] No wrist detected within {WRIST_TIMEOUT} seconds. Resetting...")
				current_state = STATE_WAITING_FOR_FACE
				saved_mask_status = None
				saved_mask_confidence = None
				baseline_temp = None
				temp_readings = []
				continue
			
			# Set baseline on first reading
			if baseline_temp is None and object_temp > 0:
				baseline_temp = object_temp
				print(f"[INFO] Baseline temperature set: {baseline_temp:.1f}°C")
			
			# Track temperature readings for stability
			if object_temp > 0:
				temp_readings.append(object_temp)
				if len(temp_readings) > 5:  # Keep last 5 readings
					temp_readings.pop(0)
			
			# Check if valid wrist is present:
			# 1. Temperature is above minimum human temp
			# 2. Temperature jumped significantly from baseline (at least 5°C increase)
			# 3. Temperature is stable (last few readings are similar)
			temp_increase = object_temp - baseline_temp if baseline_temp else 0
			temp_stable = len(temp_readings) >= 3 and (max(temp_readings[-3:]) - min(temp_readings[-3:])) < 1.0
			
			wrist_detected = (object_temp >= TEMP_MIN and 
			                 temp_increase >= 5.0 and 
			                 temp_stable)
			
			if wrist_detected:
				# Valid wrist temperature detected!
				mask_label = "Mask" if saved_mask_status else "No Mask"
				
				# Determine pass/fail
				result, result_color, result_message = check_pass_fail(saved_mask_status, object_temp)
				
				timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
				print(f"\n{'='*60}")
				print(f"[{timestamp}]")
				print(f"  STAGE 1: {mask_label} ({saved_mask_confidence:.2f}%)")
				print(f"  STAGE 2: Wrist Temperature: {object_temp:.2f}°C ({object_temp * 9/5 + 32:.2f}°F)")
				print(f"  Ambient: {ambient_temp:.2f}°C")
				print(f"  Temperature increase from baseline: +{temp_increase:.1f}°C")
				print(f"")
				print(f"  >>> RESULT: {result} <<<")
				print(f"  Message: {result_message}")
				print(f"{'='*60}")
				
				# Save image if enabled
				if args["save_images"]:
					# Create an info overlay
					info_frame = frame_rgb.copy()
					
					# Add text overlay
					cv2.putText(info_frame, f"MASK: {mask_label} ({saved_mask_confidence:.1f}%)", 
						(10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, result_color, 2)
					cv2.putText(info_frame, f"WRIST TEMP: {object_temp:.1f}C ({object_temp * 9/5 + 32:.1f}F)", 
						(10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, result_color, 2)
					cv2.putText(info_frame, f"RESULT: {result}", 
						(10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, result_color, 3)
					cv2.putText(info_frame, result_message, 
						(10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, result_color, 2)
					
					filename = f"{output_dir}/{result}_{frame_count:05d}.jpg"
					cv2.imwrite(filename, info_frame)
					print(f"[INFO] Saved detection image: {filename}")
					frame_count += 1
				
				# Move to complete state
				current_state = STATE_COMPLETE
			else:
				# Still waiting for wrist
				remaining = WRIST_TIMEOUT - elapsed
				if int(elapsed * 2) % 2 == 0 and elapsed % 0.5 < 0.1:  # Print every second
					status = f"Temp: {object_temp:.1f}°C"
					if baseline_temp:
						status += f" (baseline: {baseline_temp:.1f}°C, Δ+{temp_increase:.1f}°C)"
					print(f"[INFO] Waiting for wrist... ({remaining:.0f}s) {status}")
		
		elif current_state == STATE_COMPLETE:
			# Determine if this was a PASS result that needs cooldown
			# (no cooldown for FAIL results - they can try again immediately)
			needs_cooldown = saved_mask_status is not None and saved_mask_status == True
			
			if needs_cooldown:
				# Successful screening - enforce cooldown period
				print(f"\n[INFO] Screening complete. Cooldown period: {COOLDOWN_PERIOD} seconds...")
				print("[INFO] Please clear the area for the next person.")
				
				# Show countdown
				for remaining in range(int(COOLDOWN_PERIOD), 0, -1):
					print(f"[INFO] Ready for next person in {remaining} seconds...")
					time.sleep(1)
			else:
				# Failed screening - short delay only
				print("\n[INFO] Screening complete. Ready for next person in 2 seconds...")
				time.sleep(2)
			
			# Reset for next person
			current_state = STATE_WAITING_FOR_FACE
			saved_mask_status = None
			saved_mask_confidence = None
			baseline_temp = None
			temp_readings = []
			print("\n[INFO] Ready for next person. Please show your face to the camera.")

		# Small delay
		time.sleep(0.1)

except KeyboardInterrupt:
	print("\n[INFO] Stopping...")

# do a bit of cleanup
picam2.stop()
print("[INFO] Cleanup complete")
