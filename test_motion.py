#!/usr/bin/env python3
"""
Motion Detection Test Script
Tests screenshot capture, motion detection, and optionally Gemini proximity detection.
Saves image pairs when motion is detected for threshold tuning.
"""

import os
import sys
import json
import time
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

try:
    import cv2
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    import google.generativeai as genai_vision  # Old API for vision/text
    from mss import mss
except ImportError as e:
    print(f"❌ Missing dependency: {e}")
    print("   Run: pip install -r requirements.txt")
    sys.exit(1)


# ============================================================================
# LOGGING
# ============================================================================

def log(message, level="INFO"):
    """Print timestamped log message"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

    # Color codes for different log levels
    colors = {
        "INFO": "\033[37m",      # White
        "SUCCESS": "\033[92m",   # Green
        "WARNING": "\033[93m",   # Yellow
        "ERROR": "\033[91m",     # Red
        "DEBUG": "\033[94m",     # Blue
        "MOTION": "\033[95m",    # Magenta
    }

    reset = "\033[0m"
    color = colors.get(level, colors["INFO"])

    print(f"{color}[{timestamp}] [{level:7}] {message}{reset}")


# ============================================================================
# CONFIGURATION
# ============================================================================

def load_config():
    """Load configuration from config.json"""
    config_path = Path(__file__).parent / "config.json"

    if not config_path.exists():
        log("config.json not found, using defaults", "WARNING")
        return {
            "motion_check_interval": 1,
            "motion_threshold": 5000,
            "capture_region": None,
            "screen_number": 0
        }

    with open(config_path, 'r') as f:
        config = json.load(f)

    return config


# ============================================================================
# SCREEN CAPTURE (from scare.py)
# ============================================================================

def capture_screen(config):
    """Capture screenshot of specified region or full screen (cross-platform using mss)"""
    try:
        with mss() as sct:
            region = config.get('capture_region')
            monitor_num = config.get('screen_number', 0)

            if region:
                # Custom region: {"top": y1, "left": x1, "width": w, "height": h}
                log(f"  Capturing region: {region}", "DEBUG")

                # If region is [x1, y1, x2, y2], convert to mss format
                if isinstance(region, list) and len(region) == 4:
                    x1, y1, x2, y2 = region
                    monitor = {
                        "top": y1,
                        "left": x1,
                        "width": x2 - x1,
                        "height": y2 - y1
                    }
                else:
                    monitor = region
            else:
                # Capture entire monitor (1 = first monitor, 0 = all monitors)
                monitor_index = monitor_num + 1  # mss uses 1-based indexing for monitors
                log(f"  Capturing monitor #{monitor_num}", "DEBUG")
                monitor = sct.monitors[monitor_index]

            # Capture screenshot
            sct_img = sct.grab(monitor)

            # Convert mss image to PIL Image
            screenshot = Image.frombytes('RGB', sct_img.size, sct_img.rgb)

            log(f"  Screenshot captured: {screenshot.size[0]}x{screenshot.size[1]}", "DEBUG")
            return screenshot

    except Exception as e:
        log(f"Screenshot failed: {e}", "ERROR")
        import traceback
        log(f"  Traceback: {traceback.format_exc()}", "DEBUG")
        return None


# ============================================================================
# MOTION DETECTION (from scare.py)
# ============================================================================

def detect_motion(current_frame_pil, previous_frame_pil, config):
    """
    Simple motion detection using pixel difference
    Returns: (motion_detected, changed_pixels)
    """
    if previous_frame_pil is None:
        log("  No previous frame for motion detection", "DEBUG")
        return False, 0

    try:
        log("🔍 Detecting motion...", "DEBUG")

        # Convert PIL to OpenCV format
        current = cv2.cvtColor(np.array(current_frame_pil), cv2.COLOR_RGB2GRAY)
        previous = cv2.cvtColor(np.array(previous_frame_pil), cv2.COLOR_RGB2GRAY)

        # Resize for faster processing
        current = cv2.resize(current, (640, 480))
        previous = cv2.resize(previous, (640, 480))

        # Calculate difference
        diff = cv2.absdiff(current, previous)

        # Threshold
        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

        # Count changed pixels
        changed_pixels = cv2.countNonZero(thresh)

        threshold = config.get('motion_threshold', 5000)

        log(f"  Changed pixels: {changed_pixels} (threshold: {threshold})", "DEBUG")

        if changed_pixels > threshold:
            log(f"✅ Motion detected! ({changed_pixels} pixels changed)", "SUCCESS")
            return True, changed_pixels
        else:
            log(f"  No significant motion", "DEBUG")
            return False, changed_pixels

    except Exception as e:
        log(f"Motion detection error: {e}", "ERROR")
        return False, 0


# ============================================================================
# GEMINI PROXIMITY CHECK (from scare.py)
# ============================================================================

def init_gemini(api_key):
    """Initialize Gemini API client"""
    if not api_key:
        log("No Gemini API key provided - skipping proximity checks", "WARNING")
        return None

    log("Initializing Gemini API client...")

    try:
        genai_vision.configure(api_key=api_key)
        log("Gemini API client initialized", "SUCCESS")
        return True
    except Exception as e:
        log(f"Failed to initialize Gemini API: {e}", "ERROR")
        return None


def check_proximity_gemini(screenshot_pil, config):
    """
    Use Gemini Vision API to check if people are within 3 feet
    Returns: "YES", "NO", or None on error
    """
    log("🔍 Checking proximity with Gemini Vision API...", "DEBUG")

    try:
        model = genai_vision.GenerativeModel('gemini-2.0-flash-exp')

        prompt = (
            "Black and white night vision doorbell camera image. "
            "Are there people visible within 3 feet of the camera? "
            "Answer ONLY: YES or NO"
        )

        log(f"  Sending image to Gemini Vision...", "DEBUG")
        start_time = time.time()

        response = model.generate_content([prompt, screenshot_pil])

        elapsed = time.time() - start_time
        result = response.text.strip().upper()

        log(f"  Gemini Vision response: '{result}' ({elapsed:.2f}s)", "DEBUG")

        if "YES" in result:
            log("✅ People detected within 3 feet!", "SUCCESS")
            return "YES"
        else:
            log("  No people within 3 feet", "DEBUG")
            return "NO"

    except Exception as e:
        log(f"Gemini Vision API error: {e}", "ERROR")
        return None


# ============================================================================
# IMAGE PAIR SAVING
# ============================================================================

def save_motion_pair(previous_frame, current_frame, changed_pixels, motion_count, output_dir):
    """
    Save a pair of images that triggered motion detection
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create comparison image with both frames side by side
    width, height = current_frame.size
    comparison = Image.new('RGB', (width * 2 + 20, height + 100), color='black')

    # Add previous and current frames
    comparison.paste(previous_frame, (0, 50))
    comparison.paste(current_frame, (width + 20, 50))

    # Add labels
    draw = ImageDraw.Draw(comparison)

    try:
        # Try to use system font
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 30)
        small_font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 20)
    except:
        try:
            font = ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc', 30)
            small_font = ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc', 20)
        except:
            font = ImageFont.load_default()
            small_font = font

    # Draw labels
    draw.text((10, 10), "PREVIOUS FRAME", fill='white', font=font)
    draw.text((width + 30, 10), "CURRENT FRAME (MOTION)", fill='yellow', font=font)

    # Draw stats at bottom
    stats_text = f"Motion #{motion_count} | Timestamp: {timestamp} | Changed Pixels: {changed_pixels:,}"
    draw.text((10, height + 60), stats_text, fill='green', font=small_font)

    # Save comparison
    filename = f"motion_{timestamp}_{motion_count:03d}_pixels{changed_pixels}.png"
    filepath = output_dir / filename
    comparison.save(filepath)

    log(f"  Saved motion pair: {filename}", "SUCCESS")

    return filepath


# ============================================================================
# MAIN TEST LOOP
# ============================================================================

def main():
    """Main test loop for motion detection"""

    log("=" * 80)
    log("🧪 MOTION DETECTION TEST SCRIPT 🧪", "MOTION")
    log("=" * 80)
    log("")

    # Load environment variables
    load_dotenv()

    # Load configuration
    config = load_config()
    log(f"Configuration:", "INFO")
    log(f"  Motion check interval: {config.get('motion_check_interval', 1)}s", "INFO")
    log(f"  Motion threshold: {config.get('motion_threshold', 5000)} pixels", "INFO")
    log(f"  Screen number: {config.get('screen_number', 0)}", "INFO")
    if config.get('capture_region'):
        log(f"  Capture region: {config.get('capture_region')}", "INFO")
    log("")

    # Create output directory for motion snapshots
    output_dir = Path("test_output/motion_snapshots")
    output_dir.mkdir(parents=True, exist_ok=True)
    log(f"Snapshot output directory: {output_dir.absolute()}", "INFO")
    log("")

    # Optional: Initialize Gemini for proximity checks
    api_key = os.getenv('GEMINI_API_KEY', '')
    gemini_enabled = False

    if api_key:
        response = input("Enable Gemini proximity checks? (y/n): ").lower().strip()
        if response == 'y':
            gemini_enabled = init_gemini(api_key)
    else:
        log("No GEMINI_API_KEY in .env - proximity checks disabled", "WARNING")

    log("")
    log("=" * 80)
    log("KEYBOARD CONTROLS:", "INFO")
    log("  Press Ctrl+C to stop", "INFO")
    log("=" * 80)
    log("")

    # Test loop variables
    previous_screenshot = None
    motion_count = 0
    check_count = 0
    total_pixels_changed = 0
    motion_check_interval = config.get('motion_check_interval', 1)

    # Statistics tracking
    pixel_changes = []

    log("🚀 Starting motion detection test loop...", "SUCCESS")
    log("")

    try:
        while True:
            check_count += 1
            log(f"\n{'='*60}", "INFO")
            log(f"Check #{check_count}", "INFO")
            log(f"{'='*60}", "INFO")

            # Capture screenshot
            current_screenshot = capture_screen(config)

            if current_screenshot is None:
                log("Screenshot capture failed, retrying...", "WARNING")
                time.sleep(1)
                continue

            # Motion detection
            motion_detected, changed_pixels = detect_motion(
                current_screenshot,
                previous_screenshot,
                config
            )

            # Track statistics
            if changed_pixels > 0:
                pixel_changes.append(changed_pixels)
                total_pixels_changed += changed_pixels

            # Calculate statistics
            if pixel_changes:
                avg_change = sum(pixel_changes) / len(pixel_changes)
                max_change = max(pixel_changes)
                min_change = min(pixel_changes)

                log(f"📊 Statistics:", "INFO")
                log(f"   Current: {changed_pixels:,} pixels", "INFO")
                log(f"   Average: {avg_change:,.0f} pixels", "INFO")
                log(f"   Max: {max_change:,} pixels", "INFO")
                log(f"   Min: {min_change:,} pixels", "INFO")
                log(f"   Threshold: {config.get('motion_threshold', 5000):,} pixels", "INFO")

            if motion_detected and previous_screenshot is not None:
                motion_count += 1

                log("=" * 60, "MOTION")
                log(f"🚨 MOTION DETECTED! (Event #{motion_count})", "MOTION")
                log("=" * 60, "MOTION")

                # Save the image pair
                saved_path = save_motion_pair(
                    previous_screenshot,
                    current_screenshot,
                    changed_pixels,
                    motion_count,
                    output_dir
                )

                # Optional: Check proximity with Gemini
                if gemini_enabled:
                    log("Checking proximity with Gemini...", "INFO")
                    proximity_result = check_proximity_gemini(current_screenshot, config)

                    if proximity_result == "YES":
                        log("🎯 Gemini confirmed: People within 3 feet!", "SUCCESS")
                    elif proximity_result == "NO":
                        log("⚠️  Gemini says: No people within 3 feet", "WARNING")
                    else:
                        log("❌ Gemini check failed", "ERROR")

            # Store current screenshot for next iteration
            previous_screenshot = current_screenshot

            # Wait before next check
            log(f"⏳ Waiting {motion_check_interval}s before next check...", "DEBUG")
            time.sleep(motion_check_interval)

    except KeyboardInterrupt:
        log("\n\n⚠️  Test stopped by user", "WARNING")

    finally:
        log("\n" + "=" * 80, "INFO")
        log("📊 FINAL STATISTICS", "INFO")
        log("=" * 80, "INFO")
        log(f"Total checks: {check_count}", "INFO")
        log(f"Motion events detected: {motion_count}", "INFO")

        if pixel_changes:
            avg_change = sum(pixel_changes) / len(pixel_changes)
            log(f"Average pixel change: {avg_change:,.0f}", "INFO")
            log(f"Max pixel change: {max(pixel_changes):,}", "INFO")
            log(f"Min pixel change: {min(pixel_changes):,}", "INFO")
            log(f"Current threshold: {config.get('motion_threshold', 5000):,} pixels", "INFO")

            # Suggest threshold adjustments
            if motion_count == 0 and avg_change < config.get('motion_threshold', 5000) / 2:
                suggested = int(avg_change * 1.5)
                log(f"\n💡 SUGGESTION: Lower threshold to ~{suggested:,} pixels", "WARNING")
            elif motion_count > check_count * 0.5:
                suggested = int(max(pixel_changes) * 0.5)
                log(f"\n💡 SUGGESTION: Raise threshold to ~{suggested:,} pixels", "WARNING")

        log(f"\nSnapshot images saved to: {output_dir.absolute()}", "SUCCESS")
        log("👋 Test complete!", "INFO")


if __name__ == "__main__":
    main()
