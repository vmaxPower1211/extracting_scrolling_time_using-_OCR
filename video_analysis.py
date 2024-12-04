import cv2
import pytesseract
from pytesseract import Output
import datetime
import os
from skimage.metrics import structural_similarity as ssim
import numpy as np

# Set Tesseract executable path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Input MP4 video file
video_path = "WhatsApp Video 2024-11-26 at 22.30.10_2914e7b7 (1).mp4"  # Replace with your MP4 video file path

# Create output directory for saved images
output_dir = "tracked_images"
os.makedirs(output_dir, exist_ok=True)

# Open video file
cap = cv2.VideoCapture(video_path)

# Check if video file opened successfully
if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

# Initialize variables
frame_count = 0
fps = cap.get(cv2.CAP_PROP_FPS)  # Frames per second
interaction_data = {"pages": [], "scrolls": [], "add_to_cart": []}

prev_frame = None
last_page_saved_frame = None
scrolling = False
scroll_start_frame = 0
scroll_threshold = 20000  # Threshold for detecting scrolling
current_page_start_frame = 0
current_page_text = ""

# Helper function to convert frame number to timestamp
def timestamp_from_frame(frame, fps):
    total_seconds = frame / fps
    return str(datetime.timedelta(seconds=int(total_seconds)))

# Function to compare images using SSIM
def are_frames_similar(frame1, frame2, threshold=0.5):
    if frame1 is None or frame2 is None:
        return False
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    similarity, _ = ssim(gray1, gray2, full=True)
    return similarity > threshold

# Function to check for "add to cart" action in extracted text
def detect_add_to_cart(text):
    add_to_cart_keywords = ["add to cart", "add item", "add to basket", "cart", "add"]
    text = text.lower()
    return any(keyword in text for keyword in add_to_cart_keywords)

# Process video frame-by-frame
while True:
    ret, frame = cap.read()
    if not ret:  # End of video
        break

    frame_count += 1

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # OCR: Extract text from the frame
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    custom_config = r'--oem 3 --psm 6'
    text_data = pytesseract.image_to_data(binary, config=custom_config, output_type=Output.DICT)
    detected_text = " ".join(text_data['text']).strip()

    # Timestamp for the current frame
    timestamp = timestamp_from_frame(frame_count, fps)

    ### Page Detection ###
    if detected_text and detected_text != current_page_text:  # New screen detected
        # Save the current page's duration
        if current_page_text:  # Ensure it's not the first page
            time_spent = (frame_count - current_page_start_frame) / fps
            interaction_data["pages"].append({
                "frame_no": current_page_start_frame,
                "timestamp": timestamp_from_frame(current_page_start_frame, fps),
                "text": current_page_text,
                "time_spent_seconds": time_spent
            })

        # Save the new screen image only if it's visually different
        if last_page_saved_frame is None or not are_frames_similar(last_page_saved_frame, frame):
            image_filename = f"{output_dir}/page_{frame_count}_at_{timestamp.replace(':', '-')}.png"
            cv2.imwrite(image_filename, frame)
            last_page_saved_frame = frame

        # Update the current page info
        current_page_text = detected_text
        current_page_start_frame = frame_count

    ### Scroll Detection ###
    if prev_frame is not None:
        frame_diff = cv2.absdiff(prev_frame, gray)
        non_zero_count = np.count_nonzero(frame_diff)
        if non_zero_count > scroll_threshold:
            if not scrolling:
                # Scrolling starts
                scrolling = True
                scroll_start_frame = frame_count
        elif scrolling:
            # Scrolling ends
            scrolling = False
            scroll_time = (frame_count - scroll_start_frame) / fps
            interaction_data["scrolls"].append({
                "start_frame_no": scroll_start_frame,
                "start_timestamp": timestamp_from_frame(scroll_start_frame, fps),
                "end_frame_no": frame_count,
                "end_timestamp": timestamp,
                "scroll_time_seconds": scroll_time,
                "detected_text": current_page_text
            })

    # Check for "add to cart" in the detected text
    if detect_add_to_cart(detected_text):
        interaction_data["add_to_cart"].append({
            "frame_no": frame_count,
            "timestamp": timestamp,
            "text": detected_text
        })

    # Update previous frame for next iteration
    prev_frame = gray

# Finalize the last page duration
if current_page_text and frame_count > current_page_start_frame:
    time_spent = (frame_count - current_page_start_frame) / fps
    interaction_data["pages"].append({
        "frame_no": current_page_start_frame,
        "timestamp": timestamp_from_frame(current_page_start_frame, fps),
        "text": current_page_text,
        "time_spent_seconds": time_spent
    })

# Release video capture object
cap.release()

# Save results to a log file
log_file = "video_analysis_log.txt"
with open(log_file, "w") as f:
    # Write Page Analysis
    f.write("Pages Detected:\n")
    for page in interaction_data["pages"]:
        f.write(f"Frame No: {page['frame_no']}, Timestamp: {page['timestamp']}, "
                f"Text: {page['text']}, Time Spent (s): {page.get('time_spent_seconds', 0):.2f}\n")

    # Write Scroll Actions
    f.write("\nScroll Actions Detected:\n")
    for scroll in interaction_data["scrolls"]:
        f.write(f"Start Frame: {scroll['start_frame_no']}, Start Timestamp: {scroll['start_timestamp']}, "
                f"End Frame: {scroll['end_frame_no']}, End Timestamp: {scroll['end_timestamp']}, "
                f"Scroll Time (s): {scroll['scroll_time_seconds']:.2f}, Detected Text: {scroll['detected_text']}\n")

    # Write Add to Cart Actions
    f.write("\nAdd to Cart Actions Detected:\n")
    for add_to_cart in interaction_data["add_to_cart"]:
        f.write(f"Frame No: {add_to_cart['frame_no']}, Timestamp: {add_to_cart['timestamp']}, Text: {add_to_cart['text']}\n")

print("\nAnalysis Complete\nResults saved to video_analysis_log.txt")
