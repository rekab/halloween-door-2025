#!/usr/bin/env python3
"""
Halloween Doorbell Scare System
Continuous scare loop with motion detection and AI image generation
"""

import os
import sys
import json
import time
import random
import threading
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from io import BytesIO

# Platform detection
PLATFORM = sys.platform
log_platform = "macOS" if PLATFORM == "darwin" else "Linux" if PLATFORM == "linux" else "Unknown"

try:
    import cv2
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    import google.generativeai as genai_vision  # Old API for vision/text
    from google import genai as genai_image     # New API for image generation
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
        "SCARE": "\033[95m",     # Magenta
    }

    reset = "\033[0m"
    color = colors.get(level, colors["INFO"])

    print(f"{color}[{timestamp}] [{level:7}] {message}{reset}")


# ============================================================================
# CONFIGURATION
# ============================================================================

def load_config():
    """Load configuration from config.json"""
    log("Loading configuration from config.json...")

    config_path = Path(__file__).parent / "config.json"

    if not config_path.exists():
        log("config.json not found, creating default...", "WARNING")
        default_config = {
            "debug_mode": True,
            "motion_check_interval": 2,
            "motion_threshold": 5000,
            "scare_loop_interval": 3,
            "max_images_per_group": 10,
            "cooldown_duration": 5,
            "output_dir": "static/generated",
            "status_file": "static/status.json",
            "capture_region": None,
            "screen_number": 0
        }

        with open(config_path, 'w') as f:
            json.dump(default_config, f, indent=2)

        log(f"Created default config at {config_path}", "SUCCESS")
        return default_config

    with open(config_path, 'r') as f:
        config = json.load(f)

    log(f"Configuration loaded successfully", "SUCCESS")
    log(f"  Debug mode: {config.get('debug_mode', False)}", "DEBUG")
    log(f"  Motion check interval: {config.get('motion_check_interval')}s", "DEBUG")
    log(f"  Scare loop interval: {config.get('scare_loop_interval')}s", "DEBUG")
    log(f"  Max images per group: {config.get('max_images_per_group')}", "DEBUG")
    log(f"  Cooldown duration: {config.get('cooldown_duration')}s", "DEBUG")

    return config


# ============================================================================
# GEMINI API
# ============================================================================

def init_gemini(api_key):
    """Initialize Gemini API client (for vision/text API)"""
    if not api_key:
        log("No Gemini API key provided - running in placeholder-only mode", "WARNING")
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
    Use Gemini Vision API to check if people are within 10 feet
    Returns: "YES", "NO", or None on error
    """
    log("🔍 Checking proximity with Gemini Vision API...", "DEBUG")

    try:
        model = genai_vision.GenerativeModel('gemini-2.0-flash-exp')

        prompt = (
            "Are there people visible within 10 feet of the camera AND facing toward the camera? "
            "Answer ONLY: YES or NO"
        )

        log(f"  Sending image to Gemini Vision...", "DEBUG")
        start_time = time.time()

        response = model.generate_content([prompt, screenshot_pil])

        elapsed = time.time() - start_time
        result = response.text.strip().upper()

        log(f"  Gemini Vision response: '{result}' ({elapsed:.2f}s)", "DEBUG")

        if "YES" in result:
            log("✅ People detected within 10 feet!", "SUCCESS")
            return "YES"
        else:
            log("  No people within 10 feet", "DEBUG")
            return "NO"

    except Exception as e:
        log(f"Gemini Vision API error: {e}", "ERROR")
        return None


def generate_scary_image_gemini(screenshot_pil, config):
    """
    Generate scary ghost image using Gemini Nano Banana
    Returns: PIL Image or None on error
    """
    log("👻 Generating scary image with Nano Banana...", "SCARE")

    try:
        # Initialize Gemini client for image generation
        client = genai_image.Client()

        # Randomly select a scary character prompt
        prompts = [
            # Zombie
            "This is a black and white doorbell camera image showing people at a front "
            "door. Add 2-3 friendly Halloween zombies positioned directly behind the "
            "people, as if they're trick-or-treating too. "
            ""
            "The zombies should have: pale makeup, disheveled hair, worn vintage clothing, "
            "sleepy expressions with half-closed eyes, slow shuffling poses with arms "
            "slightly extended forward in a classic zombie stance. "
            ""
            "Position the zombies directly behind the people - close enough that they "
            "appear to be standing right behind them, not far in the distance. The zombies "
            "should be clearly visible and prominent in the image, emerging from the shadows "
            "immediately behind where the people stand. "
            ""
            "Keep the people in the foreground completely unchanged. Only add the costumed "
            "zombies directly behind them. Maintain the grainy black and white night vision "
            "camera quality.",

            # Grim Reaper
            "This is a black and white night vision doorbell camera image showing people "
            "at a front door. Using a low-angle perspective that makes the figure more "
            "imposing, add a single grim reaper looming directly behind the people. "
            ""
            "The grim reaper should have: a tall, towering presence (7-8 feet tall), "
            "flowing tattered black hooded cloak with deep shadows concealing the face, "
            "skeletal hands with long bony fingers emerging from the sleeves, holding a "
            "large scythe with a curved blade that catches slight moonlight, the figure "
            "standing perfectly still in a menacing pose with the hood tilted slightly "
            "downward as if watching the people. "
            ""
            "Position the reaper directly behind the people - close enough to appear as if "
            "it's standing right behind them, towering over them. The figure should be "
            "clearly visible and prominent, with details like the skeletal hands, scythe "
            "blade, and hooded silhouette clearly visible in the darkness immediately behind "
            "the people. "
            ""
            "Keep the people in the foreground COMPLETELY UNCHANGED - preserve their "
            "exact appearance, positions, and lighting. Only add the grim reaper directly "
            "behind them. Maintain the grainy, high-contrast black and white night vision "
            "camera quality.",

            # Ghost
            "This is a black and white night vision doorbell camera image showing people "
            "at a front door. Using the same fixed camera perspective as the doorbell, "
            "add a creepy female ghost figure positioned directly behind the people. "
            ""
            "The ghost should have: long, stringy black hair hanging down to conceal "
            "most of the face, pale white skin with an almost translucent quality, "
            "wearing a tattered white dress or nightgown that appears to float slightly, "
            "thin skeletal arms hanging at her sides, head tilted at an unnatural angle, "
            "one eye barely visible through the hair showing a hollow, dead stare, slight "
            "ethereal glow around the edges of the figure creating an otherworldly "
            "presence. "
            ""
            "Position the ghost directly behind the people - close enough that she appears "
            "to be standing right behind them, not far in the distance. The figure should "
            "be prominent and clearly visible, emerging from the darkness immediately behind "
            "where the people stand. The pale dress and skin should contrast dramatically "
            "against the dark background, creating a chilling \"she's right behind them\" "
            "effect. "
            ""
            "Keep the people in the foreground COMPLETELY UNCHANGED - do not alter their "
            "appearance, positions, clothing, or lighting in any way. Only add the ghost "
            "directly behind them. Maintain the grainy black and white night vision doorbell "
            "camera aesthetic.",

            # Demon
            "This is a black and white doorbell camera image showing people at a front "
            "door. Add a figure in a demon costume positioned directly behind the people, "
            "as if attending a Halloween party. "
            ""
            "The costumed figure should have: theatrical face paint or mask with horns, "
            "a dark dramatic outfit, standing in a theatrical pose with arms slightly "
            "raised, appearing mysterious and spooky but not threatening. "
            ""
            "Position the figure directly behind the people - close enough to appear as if "
            "standing right behind them. The costumed figure should be clearly visible and "
            "prominent, with details of the horns and costume visible in the darkness "
            "immediately behind where the people stand. "
            ""
            "Keep the people in the foreground completely unchanged. Only add the "
            "costumed figure directly behind them. Maintain the black and white night "
            "vision camera quality.",
        ]
        prompt = random.choice(prompts)

        log(f"  Prompt: {prompt[:80]}...", "DEBUG")
        log(f"  Sending to Nano Banana (this takes ~12 seconds)...", "DEBUG")

        start_time = time.time()

        # Use gemini-2.5-flash-image model for image generation
        response = client.models.generate_content(
            model="gemini-2.5-flash-image",
            contents=[prompt, screenshot_pil],
        )

        elapsed = time.time() - start_time

        log(f"  Nano Banana completed in {elapsed:.2f}s", "SUCCESS")
        log("  Extracting generated image...", "DEBUG")

        # Extract image from response
        if hasattr(response, 'candidates') and len(response.candidates) > 0:
            candidate = response.candidates[0]

            if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                for part in candidate.content.parts:
                    # Check for text response
                    if hasattr(part, 'text') and part.text is not None:
                        log(f"  API response text: {part.text}", "DEBUG")

                    # Extract generated image
                    if hasattr(part, 'inline_data') and part.inline_data is not None:
                        if hasattr(part.inline_data, 'data'):
                            log("✅ Found generated image in response!", "SUCCESS")
                            image_bytes = part.inline_data.data
                            generated_image = Image.open(BytesIO(image_bytes))
                            log(f"  Generated image size: {generated_image.size}", "SUCCESS")
                            return generated_image

        # If we got here, no image was found
        log("⚠️  No image data found in response", "WARNING")
        log("  Returning original image as fallback", "WARNING")
        return screenshot_pil

    except Exception as e:
        log(f"Nano Banana generation error: {e}", "ERROR")
        import traceback
        log(f"  Traceback: {traceback.format_exc()}", "DEBUG")
        return None


# ============================================================================
# PLACEHOLDER IMAGE GENERATION
# ============================================================================

def create_placeholder_image(screenshot_pil, image_count, config):
    """Create a test placeholder instead of calling Nano Banana"""
    log(f"🧪 Creating placeholder image #{image_count + 1}...", "DEBUG")

    try:
        # Get dimensions from screenshot
        width, height = screenshot_pil.size

        log(f"  Screenshot dimensions: {width}x{height}", "DEBUG")

        # Create scary red/black image
        img = Image.new('RGB', (width, height), color='black')
        draw = ImageDraw.Draw(img)

        # Draw red gradient background
        for y in range(height):
            darkness = int(128 * (y / height))
            draw.line([(0, y), (width, y)], fill=(darkness, 0, 0))

        # Add scary text
        text_lines = [
            f"SCARE #{image_count + 1}",
            "",
            "👻",
            "",
            "BOO!"
        ]

        try:
            # Try to use system font (cross-platform)
            font_paths = [
                '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',  # Linux
                '/System/Library/Fonts/Helvetica.ttc',  # macOS
                'C:\\Windows\\Fonts\\arial.ttf',  # Windows
            ]

            font = None
            for font_path in font_paths:
                try:
                    font = ImageFont.truetype(font_path, 80)
                    small_font = ImageFont.truetype(font_path, 40)
                    break
                except:
                    continue

            if font is None:
                raise Exception("No system fonts found")

        except:
            log("  Could not load system font, using default", "WARNING")
            font = ImageFont.load_default()
            small_font = font

        # Draw text centered
        y_pos = height // 3
        for line in text_lines:
            # Get text bbox
            bbox = draw.textbbox((0, 0), line, font=font)
            text_width = bbox[2] - bbox[0]

            x_pos = (width - text_width) // 2
            draw.text((x_pos, y_pos), line, fill='white', font=font)
            y_pos += 100

        # Add timestamp at bottom
        timestamp_text = datetime.now().strftime("%H:%M:%S")
        draw.text((20, height - 60), f"Generated: {timestamp_text}",
                 fill='gray', font=small_font)

        log(f"  Placeholder image created", "SUCCESS")
        return img

    except Exception as e:
        log(f"Failed to create placeholder: {e}", "ERROR")
        return None


# ============================================================================
# SCREEN CAPTURE
# ============================================================================

def detect_session_type():
    """
    Detect if running on Wayland or X11
    Returns: "wayland", "x11", or "unknown"
    """
    # Check XDG_SESSION_TYPE environment variable (most reliable)
    session_type = os.environ.get('XDG_SESSION_TYPE', '').lower()
    if session_type in ['wayland', 'x11']:
        return session_type

    # Fallback: Check if WAYLAND_DISPLAY is set
    if os.environ.get('WAYLAND_DISPLAY'):
        return "wayland"

    # Fallback: Check if DISPLAY is set (X11)
    if os.environ.get('DISPLAY'):
        return "x11"

    return "unknown"


def capture_screen_wayland(config):
    """
    Capture screenshot on Wayland using gnome-screenshot
    Returns: PIL Image or None
    """
    import subprocess
    import tempfile

    try:
        # Create temporary file for screenshot
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            tmp_path = tmp_file.name

        log(f"  Using Wayland capture via gnome-screenshot", "DEBUG")

        # Run gnome-screenshot to capture to temp file
        # -f = file path, -p = no effects
        result = subprocess.run(
            ['gnome-screenshot', '-f', tmp_path, '-p'],
            capture_output=True,
            timeout=5,
            check=False
        )

        if result.returncode != 0:
            log(f"gnome-screenshot failed: {result.stderr.decode()}", "ERROR")
            os.unlink(tmp_path)
            return None

        # Load the screenshot with PIL
        screenshot = Image.open(tmp_path)

        # Handle region cropping if specified
        region = config.get('capture_region')
        if region:
            if isinstance(region, list) and len(region) == 4:
                x1, y1, x2, y2 = region
                screenshot = screenshot.crop((x1, y1, x2, y2))
                log(f"  Cropped to region: {region}", "DEBUG")

        # Clean up temp file
        os.unlink(tmp_path)

        log(f"  Screenshot captured: {screenshot.size[0]}x{screenshot.size[1]}", "DEBUG")
        return screenshot

    except subprocess.TimeoutExpired:
        log("gnome-screenshot timed out", "ERROR")
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        return None
    except Exception as e:
        log(f"Wayland screenshot failed: {e}", "ERROR")
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        return None


def capture_screen(config):
    """
    Capture screenshot of specified region or full screen
    Supports both X11 (mss) and Wayland (gnome-screenshot)
    """
    # Detect session type
    session = detect_session_type()

    # Route to appropriate capture method
    if session == "wayland":
        log(f"  Detected Wayland session", "DEBUG")
        return capture_screen_wayland(config)

    elif session == "x11":
        log(f"  Detected X11 session", "DEBUG")
        # Use mss for X11 (existing code)
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
            log(f"X11 screenshot failed: {e}", "ERROR")
            import traceback
            log(f"  Traceback: {traceback.format_exc()}", "DEBUG")
            return None

    else:
        log(f"Unknown session type: {session}", "ERROR")
        log(f"  Set XDG_SESSION_TYPE environment variable to 'x11' or 'wayland'", "ERROR")
        return None


# ============================================================================
# MOTION DETECTION
# ============================================================================

def detect_motion(current_frame_pil, previous_frame_pil, config):
    """
    Simple motion detection using pixel difference
    Returns: True if motion detected, False otherwise
    """
    if previous_frame_pil is None:
        log("  No previous frame for motion detection", "DEBUG")
        return False

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
            return True
        else:
            log(f"  No significant motion", "DEBUG")
            return False

    except Exception as e:
        log(f"Motion detection error: {e}", "ERROR")
        return False


# ============================================================================
# STATUS FILE MANAGEMENT
# ============================================================================

def update_status(state, image_url, config):
    """
    Atomically update status.json
    state: "VIDEO" or "IMAGE"
    image_url: relative path like "generated/scare_001.png" or None
    """
    log(f"📝 Updating status: {state}", "DEBUG")

    try:
        status_file = Path(config['status_file'])
        status_file.parent.mkdir(parents=True, exist_ok=True)

        status_data = {
            "state": state,
            "image_url": image_url,
            "timestamp": time.time()
        }

        log(f"  State: {state}", "DEBUG")
        if image_url:
            log(f"  Image URL: {image_url}", "DEBUG")

        # Write to temp file first (atomic write)
        temp_file = status_file.with_suffix('.tmp')
        with open(temp_file, 'w') as f:
            json.dump(status_data, f, indent=2)

        # Atomic rename
        temp_file.replace(status_file)

        log(f"  status.json updated successfully", "SUCCESS")

    except Exception as e:
        log(f"Failed to update status: {e}", "ERROR")


def save_image(image_pil, filename, config):
    """
    Save generated image to output directory
    Returns: full path to saved image
    """
    log(f"💾 Saving image: {filename}", "DEBUG")

    try:
        output_dir = Path(config['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)

        filepath = output_dir / filename

        image_pil.save(filepath)

        file_size = filepath.stat().st_size
        log(f"  Image saved: {filepath} ({file_size} bytes)", "SUCCESS")

        return filepath

    except Exception as e:
        log(f"Failed to save image: {e}", "ERROR")
        return None


# ============================================================================
# SCARE SEQUENCE
# ============================================================================

def trigger_scare_sequence(initial_screenshot, config, gemini_enabled):
    """
    Main scare sequence - continuously generate images until people leave
    """
    log("=" * 80, "SCARE")
    log("🎃 SCARE SEQUENCE TRIGGERED!", "SCARE")
    log("=" * 80, "SCARE")

    image_count = 0
    max_images = config.get('max_images_per_group', 10)
    scare_interval = config.get('scare_loop_interval', 3)
    debug_mode = config.get('debug_mode', False)

    log(f"Configuration:", "SCARE")
    log(f"  Max images: {max_images}", "SCARE")
    log(f"  Interval: {scare_interval}s", "SCARE")
    log(f"  Debug mode: {debug_mode}", "SCARE")

    while image_count < max_images:
        log(f"\n--- Scare Loop Iteration #{image_count + 1} ---", "SCARE")

        # STAGE 1: HANDS ANIMATION (only on first iteration)
        # Frontend will handle timing - backend doesn't sleep
        if image_count == 0:
            log("👋 STAGE 1: Starting hands animation on frontend...", "SCARE")
            update_status("HANDS_ANIMATION", None, config)

        # Capture fresh screenshot immediately
        log("📸 Capturing fresh screenshot...", "DEBUG")
        screenshot = capture_screen(config)

        if screenshot is None:
            log("Screenshot failed, exiting scare loop", "ERROR")
            break

        # Check if people still present (skip on first iteration)
        if image_count > 0:
            if gemini_enabled and not debug_mode:
                log("Checking if people are still present...", "DEBUG")
                proximity = check_proximity_gemini(screenshot, config)

                if proximity != "YES":
                    log(f"👋 People left after {image_count} image(s)", "SUCCESS")
                    log(f"Scare sequence complete!", "SCARE")
                    break
            else:
                log("  Skipping proximity check (debug mode or no API)", "DEBUG")

        # STAGE 2: DETECTING - Save snapshot and show "Evil Spirit Detector" overlay
        log("🔍 STAGE 2: Setting detecting screen...", "SCARE")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        snapshot_filename = f"detecting_snapshot_{timestamp}.png"
        snapshot_path = save_image(screenshot, snapshot_filename, config)

        if snapshot_path is None:
            log("Failed to save snapshot, exiting scare loop", "ERROR")
            break

        snapshot_relative = f"generated/{snapshot_filename}"
        update_status("DETECTING", snapshot_relative, config)
        log("📷 Snapshot saved and detecting screen ready", "DEBUG")

        # STAGE 3: Generate image immediately (AI runs while frontend shows animations)
        log("🎨 STAGE 3: Starting AI generation...", "SCARE")
        if debug_mode:
            log("🧪 TEST MODE: Creating placeholder image", "SCARE")
            scary_image = create_placeholder_image(screenshot, image_count, config)
        else:
            if gemini_enabled:
                log(f"🎨 Generating scary image #{image_count + 1} with Nano Banana...", "SCARE")
                scary_image = generate_scary_image_gemini(screenshot, config)
            else:
                log("No API key, using placeholder", "WARNING")
                scary_image = create_placeholder_image(screenshot, image_count, config)

        if scary_image is None:
            log("Image generation failed, exiting scare loop", "ERROR")
            break

        # Save generated image
        scare_filename = f"scare_{timestamp}_{image_count:02d}.png"
        saved_path = save_image(scary_image, scare_filename, config)

        if saved_path is None:
            log("Failed to save image, exiting scare loop", "ERROR")
            break

        # Update status to show generated image
        relative_path = f"generated/{scare_filename}"
        update_status("IMAGE", relative_path, config)

        log(f"✅ Scare image #{image_count + 1} displayed!", "SCARE")

        image_count += 1

        # Wait before checking again
        log(f"⏳ Waiting {scare_interval}s before next check...", "DEBUG")
        time.sleep(scare_interval)

    # Exit scare loop
    log("=" * 80, "SCARE")
    log(f"🏁 Scare sequence complete! Total images: {image_count}", "SCARE")
    log("=" * 80, "SCARE")

    # Return to video
    log("📹 Returning to video state...", "DEBUG")
    update_status("VIDEO", None, config)

    return image_count


# ============================================================================
# KEYBOARD INPUT (NON-BLOCKING)
# ============================================================================

# Global variable for keyboard input
keyboard_input = None
keyboard_lock = threading.Lock()

def keyboard_listener():
    """Background thread to listen for keyboard input"""
    global keyboard_input

    while True:
        try:
            char = input()
            with keyboard_lock:
                keyboard_input = char.lower().strip()
        except:
            break

def get_keyboard_input():
    """Get keyboard input without blocking"""
    global keyboard_input

    with keyboard_lock:
        char = keyboard_input
        keyboard_input = None

    return char


# ============================================================================
# MAIN LOOP
# ============================================================================

def main():
    """Main detection and scare loop"""

    log("=" * 80)
    log("🎃 HALLOWEEN DOORBELL SCARE SYSTEM 🎃", "SCARE")
    log("=" * 80)
    log("")
    log(f"Platform: {log_platform} ({PLATFORM})", "INFO")

    # Detect and log session type
    session_type = detect_session_type()
    if session_type == "wayland":
        log(f"Display server: Wayland (using gnome-screenshot)", "INFO")
    elif session_type == "x11":
        log(f"Display server: X11 (using mss)", "INFO")
    else:
        log(f"Display server: Unknown ({session_type})", "WARNING")
    log("")

    # Load environment variables from .env file
    log("Loading environment variables from .env...")
    load_dotenv()

    # Load configuration
    config = load_config()

    # Initialize Gemini with API key from environment
    api_key = os.getenv('GEMINI_API_KEY', '')
    if api_key:
        log("API key loaded from environment", "SUCCESS")
        # Set GOOGLE_API_KEY for genai.Client() API
        os.environ['GOOGLE_API_KEY'] = api_key
    else:
        log("No GEMINI_API_KEY found in .env file", "WARNING")
    gemini_enabled = init_gemini(api_key)

    # Create output directory
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    log(f"Output directory: {output_dir.absolute()}", "DEBUG")

    # Initialize status file
    log("Initializing status file...")
    update_status("VIDEO", None, config)

    # Start keyboard listener thread
    log("Starting keyboard listener thread...", "DEBUG")
    keyboard_thread = threading.Thread(target=keyboard_listener, daemon=True)
    keyboard_thread.start()

    log("")
    log("=" * 80)
    log("KEYBOARD CONTROLS:", "INFO")
    log("  't' = Manual trigger (test scare)", "INFO")
    log("  'v' = Force back to video state", "INFO")
    log("  'q' = Quit program", "INFO")
    log("=" * 80)
    log("")

    # Main loop variables
    previous_screenshot = None
    last_scare_time = 0
    motion_check_interval = config.get('motion_check_interval', 2)
    cooldown_duration = config.get('cooldown_duration', 5)

    log("🚀 Main loop started!", "SUCCESS")
    log(f"   Motion check interval: {motion_check_interval}s", "INFO")
    log(f"   Cooldown duration: {cooldown_duration}s", "INFO")
    log("")

    try:
        while True:
            # Check for keyboard input
            key = get_keyboard_input()

            if key == 'q':
                log("👋 Quit command received", "WARNING")
                break

            elif key == 'v':
                log("📹 Force video command received", "WARNING")
                update_status("VIDEO", None, config)
                continue

            elif key == 't':
                log("🧪 Manual trigger command received!", "WARNING")
                screenshot = capture_screen(config)
                if screenshot:
                    trigger_scare_sequence(screenshot, config, gemini_enabled)
                    last_scare_time = time.time()
                else:
                    log("Manual trigger failed - no screenshot", "ERROR")
                continue

            # Check cooldown
            time_since_last_scare = time.time() - last_scare_time

            if time_since_last_scare < cooldown_duration:
                remaining = cooldown_duration - time_since_last_scare
                # Only log every 2 seconds to avoid spam
                if int(remaining) % 2 == 0:
                    log(f"⏸️  Cooldown: {remaining:.1f}s remaining...", "DEBUG")
                time.sleep(0.5)
                continue

            # Capture screenshot
            current_screenshot = capture_screen(config)

            if current_screenshot is None:
                log("Screenshot capture failed, retrying...", "WARNING")
                time.sleep(1)
                continue

            # Motion detection
            motion_detected = detect_motion(current_screenshot, previous_screenshot, config)
            previous_screenshot = current_screenshot

            if not motion_detected:
                time.sleep(motion_check_interval)
                continue

            # Motion detected - check proximity
            log("🚨 Motion detected! Checking proximity...", "WARNING")

            if gemini_enabled and not config.get('debug_mode'):
                proximity_result = check_proximity_gemini(current_screenshot, config)

                if proximity_result != "YES":
                    log("No people within 10 feet, ignoring motion", "DEBUG")
                    time.sleep(motion_check_interval)
                    continue
            else:
                log("Skipping proximity check (debug mode or no API)", "DEBUG")
                # In debug mode, any motion triggers scare

            # Trigger scare sequence!
            trigger_scare_sequence(current_screenshot, config, gemini_enabled)
            last_scare_time = time.time()

            # Brief pause before resuming
            time.sleep(motion_check_interval)

    except KeyboardInterrupt:
        log("\n\n⚠️  Keyboard interrupt received", "WARNING")

    except Exception as e:
        log(f"\n\n❌ Unexpected error: {e}", "ERROR")
        import traceback
        traceback.print_exc()

    finally:
        log("\n" + "=" * 80, "INFO")
        log("🛑 Shutting down scare system...", "WARNING")
        log("=" * 80, "INFO")

        # Return to video state
        update_status("VIDEO", None, config)
        log("✅ Status reset to VIDEO", "SUCCESS")
        log("👋 Goodbye!", "INFO")


if __name__ == "__main__":
    main()
