# 🎃 Halloween Doorbell Scare System

A Python-based Halloween scare system that displays looping creepy hands video
on a projector, then generates AI-modified halloween images continuously of
trick-or-treaters until they leave.


[![Video of the project in action](https://img.youtube.com/vi/-9dDK1ihY8A/0.jpg)](https://www.youtube.com/watch?v=-9dDK1ihY8A)

## 🎯 How It Works

1. **Default State**: Creepy hands video loops on projector
2. **Motion Detection**: System detects movement via screen capture
3. **Proximity Check**: Gemini Vision API confirms people are within 3 feet
4. **Continuous Scare Loop**:
   - Take screenshot
   - Check if people still present
   - Generate scary ghost image with Nano Banana (~12 seconds)
   - Display image for 10 seconds
   - Repeat until people leave
5. **Return to Video**: Automatically returns to video when people walk away
6. **Cooldown**: 5-second pause before next group

## 🏗️ Architecture

**Two Python Processes:**
- `serve.py` - Simple HTTP server (serves static files)
- `scare.py` - Motion detection + AI generation + state management

**Communication:**
- File-based via `status.json` (atomic writes)
- Browser polls status.json every 500ms

**No Frameworks:**
- Pure Python with `http.server`
- Simple, reliable, easy to debug

## 📁 File Structure

```
halloween-door-2025/
├── scare.py              # Main detection + generation loop
├── serve.py              # HTTP server
├── config.json           # System settings
├── .env                  # Environment variables (API keys)
├── .env.example          # Template for .env
├── requirements.txt      # Dependencies
├── README.md             # This file
├── static/
│   ├── index.html        # Display page (with polling)
│   ├── creepyhands2.mp4  # Looping creepy video
│   ├── warcry.mp3        # Audio file
│   ├── status.json       # Auto-generated state file
│   └── generated/        # Auto-generated scary images
```

## ⚙️ Requirements

- **Linux ARM64** (tested), macOS, or Windows
- **Python 3.9+**
- **Google Gemini API key** (free tier works)
- **Chrome** (for viewing Nest doorbell)
- **Projector/Chromebook** (for display)
- **Display server** (Linux):
  - X11: Uses `mss` library (fast)
  - Wayland: Requires `gnome-screenshot` (see installation below)

## 🚀 Quick Start (15 Minutes)

### 1. Install Dependencies

```bash
# Install Python packages
pip install -r requirements.txt

# If using Wayland (check with: echo $XDG_SESSION_TYPE)
sudo apt install gnome-screenshot
```

### 2. Get Gemini API Key

1. Go to https://aistudio.google.com/apikey
2. Click "Create API Key"
3. Copy the key

### 3. Configure Environment Variables

Create a `.env` file in the project root:

```bash
cp .env.example .env
```

Edit `.env` and add your API key:

```env
# Gemini API Key
# Get your API key from: https://aistudio.google.com/apikey
GEMINI_API_KEY=your_actual_api_key_here
```

### 4. Configure System

Edit `config.json`:

```json
{
  "debug_mode": true,  // Start with true for testing
  ...
}
```

### 5. Test in Debug Mode (Free)

**Terminal 1 - Start Server:**
```bash
python serve.py
```

**Terminal 2 - Start Scare System:**
```bash
python scare.py
```

**Browser:**
Open `http://YOUR-MAC-IP:8000` (shown in Terminal 1)

**Test Trigger:**
In Terminal 2, press `t` to trigger a manual scare

✅ You should see placeholder images appear (fast, free testing)

### 6. Configure Projector

1. Adjust video position using on-screen controls
2. Use "Show Controls" checkbox to hide controls
3. Position viewport perfectly on your door

### 7. Switch to Production Mode

Edit `config.json`:

```json
{
  "debug_mode": false,  // Use real AI generation
  ...
}
```

Restart `scare.py` - now uses real Gemini AI!

## 🎮 Keyboard Controls

While `scare.py` is running:

- `t` - Manual trigger (test scare sequence)
- `v` - Force back to video state
- `q` - Quit program

## ⚙️ Configuration Options

### config.json

```json
{
  "debug_mode": false,           // true = free placeholders, false = real AI

  "motion_check_interval": 1,    // Seconds between motion checks
  "motion_threshold": 5000,      // Pixels changed to trigger motion

  "scare_loop_interval": 10,     // Seconds between images during scare
  "max_images_per_group": 10,    // Safety limit per group

  "cooldown_duration": 5,        // Seconds before next group

  "output_dir": "static/generated",
  "status_file": "static/status.json",

  "capture_region": null,        // [x1, y1, x2, y2] or null for full screen
  "screen_number": 0
}
```

**Note:** API key is stored in `.env` file, not in config.json

### .env File

The `.env` file contains sensitive environment variables and should **never** be committed to git.

**Required Variables:**
```env
# Gemini API Key (required for AI generation)
# Get from: https://aistudio.google.com/apikey
GEMINI_API_KEY=your_actual_api_key_here
```

**Setup:**
1. Copy `.env.example` to `.env`
2. Add your actual API key
3. The `.env` file is already in `.gitignore` for security

**Note:** The system uses `python-dotenv` to load these variables automatically when `scare.py` runs.

### Recommended Settings

**For Testing:**
```json
{
  "debug_mode": true,
  "motion_check_interval": 2,
  "scare_loop_interval": 5,
  "max_images_per_group": 3
}
```

**For Production (Halloween Night):**
```json
{
  "debug_mode": false,
  "motion_check_interval": 1,
  "scare_loop_interval": 10,
  "max_images_per_group": 10
}
```

## 🔧 Advanced Setup

### Set Capture Region (Faster Processing)

1. Find your Nest doorbell window coordinates
2. Edit `config.json`:

```json
{
  "capture_region": [0, 0, 1920, 1080]  // [x1, y1, x2, y2]
}
```

### Find Your IP Address

**Linux:**
```bash
hostname -I
# or
ip addr show | grep "inet " | grep -v 127.0.0.1
```

**macOS:**
```bash
ifconfig | grep "inet " | grep -v 127.0.0.1
```

Look for something like `192.168.1.100`

### Linux ARM64 Specific Tips

**Font issues?** Install DejaVu fonts:
```bash
sudo apt-get install fonts-dejavu-core
```

**Display server check:**
```bash
echo $XDG_SESSION_TYPE  # Should show 'x11' or 'wayland'
```

**Test screen capture:**
```bash
python3 -c "from mss import mss; import pprint; pprint.pprint(mss().monitors)"
```

## 📊 Cost Estimation

### Debug Mode (Placeholder Images)
- **Cost**: $0
- **Speed**: ~4 seconds per image
- **Use for**: Testing positioning, layout, timing

### Production Mode (Real AI)
- **Gemini Vision** (proximity check): ~$0.0003 per check
- **Nano Banana** (image generation): ~$0.0015 per image
- **Per group** (3 images avg): ~$0.005
- **100 groups**: ~$0.50 total

**Recommendation**: Test in debug mode until everything works perfectly!

## 🧪 Testing Checklist

### Phase 1: Server Test
- [ ] `python serve.py` starts without errors
- [ ] Browser loads video page at `http://YOUR-IP:8000`
- [ ] Video loops smoothly
- [ ] Controls work (resize, move, toggle border)

### Phase 2: Placeholder Test
- [ ] `config.json` has `debug_mode: true`
- [ ] `python scare.py` starts without errors
- [ ] Press `t` in terminal
- [ ] Placeholder image appears in browser
- [ ] Multiple images cycle every 3 seconds
- [ ] Press `v` to return to video
- [ ] Video resumes smoothly

### Phase 3: Motion Detection Test
- [ ] Wave hand in front of doorbell camera
- [ ] Check terminal for motion detection logs
- [ ] Scare sequence triggers automatically
- [ ] Cooldown prevents spam

### Phase 4: Production Test
- [ ] `config.json` has `debug_mode: false`
- [ ] `.env` file has valid `GEMINI_API_KEY`
- [ ] Proximity check works (logs show "YES" or "NO")
- [ ] AI images generate (~12 seconds)
- [ ] Images are actually scary
- [ ] Continuous loop works until people leave

### Phase 5: Halloween Night Readiness
- [ ] Projector positioned correctly
- [ ] Controls hidden (`Show Controls` unchecked)
- [ ] Audio working (if desired)
- [ ] Both Python processes running stable
- [ ] Test with real person at door
- [ ] Monitor logs for errors

## 🐛 Troubleshooting

### "No module named 'cv2'"
```bash
pip install opencv-python
```

### "No module named 'google.generativeai'"
```bash
pip install google-generativeai
```

### "Screenshot failed" or "XGetImage() failed"

**Linux:**
- **Wayland users**: Install gnome-screenshot: `sudo apt install gnome-screenshot`
- **X11 users**: No special permissions needed
- Check your session type: `echo $XDG_SESSION_TYPE`
- The system automatically detects and uses the correct capture method
- Test X11: `python3 -c "from mss import mss; print(mss().monitors)"`
- Test Wayland: `gnome-screenshot -f /tmp/test.png`

**macOS:**
- System Preferences → Security & Privacy → Screen Recording
- Grant Terminal/Python screen recording permission

**General:**
- Check that display server is running
- On startup, scare.py logs which capture method it's using

### "Gemini API error"
- Check API key in `.env` file
- Verify key is active at https://aistudio.google.com/apikey
- Ensure `.env` file exists in project root
- Check internet connection

### Video doesn't loop
- Check that `creepyhands2.mp4` is in `static/` folder
- Check browser console for errors (F12)

### Images don't appear
- Check `static/status.json` exists
- Check browser console for fetch errors
- Check `static/generated/` folder has images

### Motion too sensitive
- Increase `motion_threshold` in config.json
- Try 10000 or higher

### Motion not sensitive enough
- Decrease `motion_threshold`
- Try 2000 or lower

### Images generate too slowly
- **Expected**: 12-15 seconds per image with Nano Banana
- Use `debug_mode: true` for faster testing
- Reduce `max_images_per_group`

## 📝 Logs

The system produces **detailed, colorful logs**:

```
[2025-10-25 18:05:30.123] [INFO   ] 🚀 Main loop started!
[2025-10-25 18:05:32.456] [DEBUG  ] 🔍 Detecting motion...
[2025-10-25 18:05:32.478] [SUCCESS] ✅ Motion detected! (7834 pixels changed)
[2025-10-25 18:05:32.480] [WARNING] 🚨 Motion detected! Checking proximity...
[2025-10-25 18:05:33.891] [SUCCESS] ✅ People detected within 3 feet!
[2025-10-25 18:05:33.892] [SCARE  ] 🎃 SCARE SEQUENCE TRIGGERED!
[2025-10-25 18:05:45.234] [SUCCESS] ✅ Scare image #1 displayed!
```

**Log Levels:**
- 🔵 `DEBUG` - Detailed debugging info
- ⚪ `INFO` - General information
- 🟢 `SUCCESS` - Something worked!
- 🟡 `WARNING` - Important events
- 🔴 `ERROR` - Something failed
- 🟣 `SCARE` - Scare sequence events

## 🎭 Tips for Maximum Scares

1. **Positioning**: Test projector position during daytime
2. **Timing**: Each image takes ~16 seconds in production mode
3. **Audio**: Enable warcry.mp3 for extra scare factor
4. **Lighting**: Works best in low light (dusk/evening)
5. **Patience**: Let kids get close before triggering
6. **Safety**: Don't scare kids too young - use responsibly!

## 🔐 Security Notes

- Server runs on local network only
- No external access (unless you port forward)
- API key stored in `.env` file (not committed to git)
- `.env` file is excluded via `.gitignore`
- Generated images saved locally

## 📜 License

Use freely for Halloween fun! No warranty. Scare responsibly.

## 🎃 Happy Halloween!

Built with ❤️ and 👻 by a Python enthusiast who loves logs.

---

**Questions?** Check the logs - they're very detailed!

**Issues?** Make sure both `serve.py` and `scare.py` are running.

**Too scary?** Reduce `max_images_per_group` or increase `scare_loop_interval`.
