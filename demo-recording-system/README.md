# 🎬 Demo Recording System

Interactive presentation system for teaching and presenting the [AI-Native Application Architecture Whitepaper](../README.md).

## 🎯 What is This System?

This is a complete presentation of the AI-native architecture concepts:

- **49 interactive slides** across 12 sections
- **20.4 minutes** of content covering all 15 whitepaper chapters
- **Automatic video recording** via Playwright (WebM format)
- **Modular YAML content** - easy to customize
- **Responsive design** - works on desktop, tablet, mobile

**Use this to**:
- 📚 Present to your team or organization
- 🎓 Teach AI-native architecture concepts  
- 🎬 Record professional videos
- 🎯 Customize for your specific use case

---

## 📊 Slide Coverage

| Section | Slides | Topics |
|---------|--------|--------|
| Introduction | 3 | Paradigm shift, problems solved, what we built |
| Core Concepts | 3 | MCP services, LLM role, multiple intents |
| Architecture | 5 | Deployment, multi-turn, security, testing |
| Operations | 3 | Analytics, context, training |
| Migration | 3 | Migration path, frameworks, benefits |
| Conclusion | 4 | Organizational changes, future directions |
| *Additional* | *20* | Deep dives into each major concept |

---

## Quick Start

```bash
# 1. Setup
python setup.py

# 2. Generate HTML from YAML
python slide-config-loader.py

# 3. View in browser
open demo-slides.html

# 4. Record video
python demo-recorder.py
```

## Files

### Core System
- **`slide-config-loader.py`** - Converts modular YAML to HTML (310 lines, pure logic)
- **`presentation-template.html`** - Reusable HTML/CSS/JS template (320 lines)
- **`demo-recorder.py`** - Playwright-based automated video recording
- **`setup.py`** - Installation and environment validation

### Content (Modular YAML)
- **`slides/`** - 12 modular YAML files
  - `01-introduction.yaml` → `12-conclusion.yaml`
  - 49 slides total with diagrams, code, text
  - Easy to edit without touching code

### Generated
- **`demo-slides.html`** - Interactive HTML presentation (112 KB, responsive, self-contained)
- **`recordings/`** - Video files (WebM format, VP9 codec)

## Key Features

✨ **Split-Screen Layout**
- Horizontal split on desktop (narration on left, content on right)
- Vertical split on tablets and mobile
- Responsive design with media queries

✨ **Large Narration Formatting**
- Narration displayed as large bullet points (1.5rem font)
- Better readability and slide-like appearance
- Organized by line breaks in YAML

✨ **Professional Presentation**
- Self-contained HTML (no external dependencies)
- Keyboard navigation (← →, SPACE for play/pause)
- Progress tracking and timer
- Smooth fade-in animations

✨ **Automated Recording**
- Playwright-based browser automation
- Records to WebM format (VP9 codec, open standard)
- 1920×1080 resolution, 24 fps
- Configurable timing and options

## Command Reference

### Generate HTML
```bash
python slide-config-loader.py
```

### Record Video
```bash
# Full presentation with GUI
python demo-recorder.py

# Headless mode (no browser window)
python demo-recorder.py --headless

# Test first 5 slides
python demo-recorder.py --max-slides 5

# Custom pause between slides (3 seconds)
python demo-recorder.py --pause 3000

# Faster replay
python demo-recorder.py --slow-mo 100
```

### Setup
```bash
python setup.py
```

## Configuration (YAML)

The `slides/` directory contains 12 modular YAML files. Each file has a structure like:

```yaml
sections:
  - name: "Section Name"
    slides:
      - id: slide_1
        title: "Slide Title"
        narration: |
          • Bullet point 1
          • Bullet point 2
          • Bullet point 3
        duration: 25        # seconds
        content_type: code  # "code", "text", or "diagram"
        content: |
          Some code or content
```

**Content Types:**
- `code` - Programming code (syntax highlighted)
- `text` - Text with ASCII diagrams (monospace, pre-formatted)
- `diagram` - ASCII art diagrams (preserves formatting)

## Responsive Breakpoints

- **Desktop (≥1024px)**: Horizontal split-screen
  - Left: Title + narration bullets
  - Right: Content/code
  
- **Tablet (768-1024px)**: Vertical stacked
  - Top: Title + narration
  - Bottom: Content
  
- **Mobile (<768px)**: Compact vertical
  - Reduced font sizes
  - Hidden controls legend
  - Single column layout

## Browser View

Open `demo-slides.html` in your browser to:
- Navigate with arrow keys (← →)
- Play/Pause with SPACE
- View progress bar and timer
- Enjoy responsive layout

## Video Output

The recorder generates WebM files with:
- **Codec**: VP9 (open standard)
- **Resolution**: 1920×1080 (Full HD)
- **Framerate**: 24 fps (cinema standard)
- **Size**: ~40-70 MB per minute

## Tips

1. **Content**: Edit files in `slides/` directory - no code changes needed
2. **Formatting**: Each line in narration becomes a bullet point
3. **Code**: Use `content_type: code` for syntax-highlighted code blocks
4. **ASCII Art**: Use `content_type: diagram` for ASCII diagrams (monospace, preserves whitespace)
5. **Testing**: Use `--max-slides 3` to test quickly
6. **Recording**: `--headless` mode gives cleanest results

## Troubleshooting

**Playwright not found?**
```bash
pip install playwright
playwright install
```

**YAML parse error?**
```bash
python -c "import yaml; yaml.safe_load(open('slides/01-introduction.yaml'))"
```

**Recording fails?**
```bash
# Try without headless first
python demo-recorder.py
```

## Next Steps

1. Edit files in `slides/` directory with your content
2. Run `python slide-config-loader.py` to generate HTML
3. Open `demo-slides.html` to preview
4. Run `python demo-recorder.py` to record video
5. Post-process in video editor (add narration, music, effects)
6. Export and share!

---

**Status**: Production Ready ✅  
**Version**: 1.0  
**Last Updated**: October 27, 2025
