#!/usr/bin/env python3
"""
Image Generation Test Script
Tests Gemini Nano Banana image generation with a reference image.
"""

import os
import sys
import argparse
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from io import BytesIO

try:
    from google import genai
    from PIL import Image
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
        "GENERATE": "\033[95m",  # Magenta
    }

    reset = "\033[0m"
    color = colors.get(level, colors["INFO"])

    print(f"{color}[{timestamp}] [{level:8}] {message}{reset}")


# ============================================================================
# IMAGE GENERATION
# ============================================================================

def generate_scary_image(input_image_pil, prompt=None, model_name="gemini-2.5-flash-image"):
    """
    Generate scary image using Gemini Nano Banana
    Returns: (generated_image, response_metadata) or (None, error_message)
    """
    log(f"👻 Generating scary image with {model_name}...", "GENERATE")

    # Default prompt
    if prompt is None:
        prompt = (
            "Add a TERRIFYING ghost with hollow black eyes, gaping mouth, and pale "
            "skeletal face RIGHT BEHIND the people in this black and white doorbell "
            "image. The ghost should be very close and menacing, dramatic horror "
            "lighting with high contrast, jump scare aesthetic, nightmare fuel"
        )

    log(f"  Prompt: {prompt}", "DEBUG")
    log(f"  Model: {model_name}", "DEBUG")
    log(f"  Input image size: {input_image_pil.size}", "DEBUG")

    try:
        # Initialize Gemini client
        log("  Initializing Gemini client...", "DEBUG")
        client = genai.Client()

        log(f"  Sending request to {model_name}...", "DEBUG")
        log(f"  (This may take 10-15 seconds)", "DEBUG")

        import time
        start_time = time.time()

        # Generate content
        response = client.models.generate_content(
            model=model_name,
            contents=[prompt, input_image_pil],
        )

        elapsed = time.time() - start_time
        log(f"  Request completed in {elapsed:.2f}s", "SUCCESS")

        # Inspect response structure
        log("  Inspecting response structure...", "DEBUG")
        log(f"    Type: {type(response)}", "DEBUG")
        log(f"    Has candidates: {hasattr(response, 'candidates')}", "DEBUG")

        if hasattr(response, 'candidates') and len(response.candidates) > 0:
            log(f"    Candidates count: {len(response.candidates)}", "DEBUG")
            candidate = response.candidates[0]
            log(f"    Candidate type: {type(candidate)}", "DEBUG")
            log(f"    Has content: {hasattr(candidate, 'content')}", "DEBUG")

            if hasattr(candidate, 'content'):
                content = candidate.content
                log(f"    Content type: {type(content)}", "DEBUG")
                log(f"    Has parts: {hasattr(content, 'parts')}", "DEBUG")

                if hasattr(content, 'parts'):
                    log(f"    Parts count: {len(content.parts)}", "DEBUG")

                    # Iterate through parts to find image
                    for i, part in enumerate(content.parts):
                        log(f"    Part {i} type: {type(part)}", "DEBUG")
                        log(f"    Part {i} has text: {hasattr(part, 'text')}", "DEBUG")
                        log(f"    Part {i} has inline_data: {hasattr(part, 'inline_data')}", "DEBUG")

                        # Extract text response
                        if hasattr(part, 'text') and part.text is not None:
                            log(f"    Part {i} text: {part.text}", "INFO")

                        # Extract generated image
                        if hasattr(part, 'inline_data') and part.inline_data is not None:
                            log(f"✅ Found generated image in part {i}!", "SUCCESS")
                            log(f"    Inline data type: {type(part.inline_data)}", "DEBUG")
                            log(f"    Has data attribute: {hasattr(part.inline_data, 'data')}", "DEBUG")

                            if hasattr(part.inline_data, 'data'):
                                image_bytes = part.inline_data.data
                                log(f"    Image bytes length: {len(image_bytes)}", "DEBUG")

                                # Convert bytes to PIL Image
                                generated_image = Image.open(BytesIO(image_bytes))
                                log(f"    Generated image size: {generated_image.size}", "SUCCESS")
                                log(f"    Generated image mode: {generated_image.mode}", "SUCCESS")

                                # Return success
                                metadata = {
                                    "elapsed_time": elapsed,
                                    "model": model_name,
                                    "prompt": prompt,
                                    "input_size": input_image_pil.size,
                                    "output_size": generated_image.size,
                                }

                                return generated_image, metadata

                    log("⚠️  No inline_data found in any parts", "WARNING")
                else:
                    log("❌ Content has no parts attribute", "ERROR")
            else:
                log("❌ Candidate has no content attribute", "ERROR")
        else:
            log("❌ Response has no candidates", "ERROR")

        # If we got here, no image was found
        return None, "No image data found in response"

    except Exception as e:
        log(f"❌ Image generation error: {e}", "ERROR")
        import traceback
        log(f"    Traceback: {traceback.format_exc()}", "DEBUG")
        return None, str(e)


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main test function"""

    parser = argparse.ArgumentParser(description="Test Gemini image generation")
    parser.add_argument("image_path", help="Path to input image file")
    parser.add_argument("--prompt", help="Custom prompt (optional)", default=None)
    parser.add_argument("--model", help="Model name", default="gemini-2.5-flash-image")
    parser.add_argument("--output", help="Output directory", default="test_output/image_generation")

    args = parser.parse_args()

    log("=" * 80)
    log("🧪 GEMINI IMAGE GENERATION TEST 🧪", "GENERATE")
    log("=" * 80)
    log("")

    # Load environment variables
    load_dotenv()
    api_key = os.getenv('GEMINI_API_KEY', '')

    if not api_key:
        log("❌ No GEMINI_API_KEY found in .env file", "ERROR")
        log("   Please create a .env file with your API key", "ERROR")
        sys.exit(1)

    log("✅ API key loaded from environment", "SUCCESS")

    # Set GOOGLE_API_KEY for genai.Client() API
    os.environ['GOOGLE_API_KEY'] = api_key
    log("")

    # Check input image exists
    image_path = Path(args.image_path)
    if not image_path.exists():
        log(f"❌ Input image not found: {image_path}", "ERROR")
        sys.exit(1)

    log(f"📷 Loading input image: {image_path}", "INFO")

    try:
        input_image = Image.open(image_path)
        log(f"   Size: {input_image.size}", "INFO")
        log(f"   Mode: {input_image.mode}", "INFO")
        log("")
    except Exception as e:
        log(f"❌ Failed to load image: {e}", "ERROR")
        sys.exit(1)

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    log(f"💾 Output directory: {output_dir.absolute()}", "INFO")
    log("")

    # Generate scary image
    log("🚀 Starting image generation...", "GENERATE")
    log("")

    generated_image, result = generate_scary_image(
        input_image,
        prompt=args.prompt,
        model_name=args.model
    )

    log("")
    log("=" * 80)

    if generated_image is not None:
        log("✅ IMAGE GENERATION SUCCESS!", "SUCCESS")
        log("=" * 80)

        # Save input image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        input_filename = f"input_{timestamp}.png"
        input_path = output_dir / input_filename
        input_image.save(input_path)
        log(f"📷 Saved input image: {input_path}", "SUCCESS")

        # Save generated image
        output_filename = f"generated_{timestamp}.png"
        output_path = output_dir / output_filename
        generated_image.save(output_path)
        log(f"👻 Saved generated image: {output_path}", "SUCCESS")

        # Create side-by-side comparison
        log("🖼️  Creating comparison image...", "INFO")

        # Resize images to same height for comparison
        max_height = max(input_image.size[1], generated_image.size[1])

        input_resized = input_image
        if input_image.size[1] != max_height:
            ratio = max_height / input_image.size[1]
            new_width = int(input_image.size[0] * ratio)
            input_resized = input_image.resize((new_width, max_height))

        generated_resized = generated_image
        if generated_image.size[1] != max_height:
            ratio = max_height / generated_image.size[1]
            new_width = int(generated_image.size[0] * ratio)
            generated_resized = generated_image.resize((new_width, max_height))

        # Create comparison
        comparison_width = input_resized.size[0] + generated_resized.size[0] + 20
        comparison = Image.new('RGB', (comparison_width, max_height + 60), color='black')

        # Paste images
        comparison.paste(input_resized, (0, 40))
        comparison.paste(generated_resized, (input_resized.size[0] + 20, 40))

        # Add labels
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(comparison)

        try:
            font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 24)
        except:
            try:
                font = ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc', 24)
            except:
                font = ImageFont.load_default()

        draw.text((10, 10), "INPUT", fill='white', font=font)
        draw.text((input_resized.size[0] + 30, 10), "GENERATED", fill='yellow', font=font)

        # Save comparison
        comparison_filename = f"comparison_{timestamp}.png"
        comparison_path = output_dir / comparison_filename
        comparison.save(comparison_path)
        log(f"📊 Saved comparison: {comparison_path}", "SUCCESS")

        # Print metadata
        log("")
        log("📋 Generation Metadata:", "INFO")
        for key, value in result.items():
            log(f"   {key}: {value}", "INFO")

        log("")
        log("🎉 Test complete! Check the output directory for results.", "SUCCESS")

    else:
        log("❌ IMAGE GENERATION FAILED!", "ERROR")
        log("=" * 80)
        log(f"Error: {result}", "ERROR")
        log("")
        log("💡 Troubleshooting tips:", "WARNING")
        log("   - Check that your API key has image generation permissions", "WARNING")
        log("   - Verify the model name is correct (gemini-2.5-flash-image)", "WARNING")
        log("   - Try with a different input image", "WARNING")
        sys.exit(1)


if __name__ == "__main__":
    main()
