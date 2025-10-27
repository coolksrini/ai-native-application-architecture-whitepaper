#!/usr/bin/env python3
"""
Automated Demo Recorder with Playwright
Records presentation slides with automatic navigation and timing
Generates demo video from AI-native architecture presentation
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime
import subprocess
import json
import logging

# Check if playwright is installed
try:
    from playwright.async_api import async_playwright, expect
except ImportError:
    print("❌ Playwright not installed")
    print("\nInstall with:")
    print("  pip install playwright")
    print("  playwright install")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DemoRecorder:
    """Automated recorder for demo presentation slides"""
    
    def __init__(self,
                 html_file: str = "../demo-slides.html",
                 output_dir: str = "recordings",
                 headless: bool = False,
                 record_video: bool = True,
                 slow_motion: int = 500):
        """
        Initialize the recorder
        
        Args:
            html_file: Path to HTML presentation file
            output_dir: Directory to save recordings
            headless: Run browser in headless mode
            record_video: Enable video recording
            slow_motion: Slow motion speed in milliseconds
        """
        self.html_file = Path(html_file)
        self.output_dir = Path(output_dir)
        self.headless = headless
        self.record_video = record_video
        self.slow_motion = slow_motion
        
        # Create output directory
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Video file path
        self.video_file = self.output_dir / f"demo_recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}.webm"
        
    async def record(self, 
                    auto_advance: bool = True,
                    slide_pause_ms: int = 2000,
                    max_slides: int = None):
        """
        Record the presentation
        
        Args:
            auto_advance: Automatically advance slides
            slide_pause_ms: Pause time between slides in milliseconds
            max_slides: Maximum slides to record (None = all)
        """
        logger.info(f"🎬 Starting demo recording...")
        logger.info(f"📄 Presentation: {self.html_file}")
        logger.info(f"🎥 Video output: {self.video_file}")
        logger.info(f"⏱️  Slide pause: {slide_pause_ms}ms")
        
        async with async_playwright() as p:
            # Launch browser with video recording
            record_video_dir = None
            if self.record_video:
                record_video_dir = str(self.output_dir)
            
            browser = await p.chromium.launch(
                headless=self.headless,
                slow_mo=self.slow_motion
            )
            
            # Create context with video recording
            context = await browser.new_context(
                record_video_dir=record_video_dir if self.record_video else None,
                viewport={"width": 1920, "height": 1080}
            )
            
            page = await context.new_page()
            
            # Navigate to presentation
            file_url = f"file://{self.html_file.resolve()}"
            await page.goto(file_url, wait_until="domcontentloaded")
            
            logger.info("✅ Presentation loaded")
            
            # Get total number of slides
            try:
                total_slides = await page.evaluate("() => document.querySelectorAll('[data-slide]').length")
                logger.info(f"📊 Total slides: {total_slides}")
            except Exception as e:
                logger.warning(f"Could not determine slide count: {e}")
                total_slides = None
            
            # Record slides
            slide_count = 0
            while True:
                slide_count += 1
                
                if max_slides and slide_count > max_slides:
                    logger.info(f"⏹️  Reached max slides limit ({max_slides})")
                    break
                
                # Get current slide info
                try:
                    current_info = await page.evaluate("""
                        () => {
                            const slides = document.querySelectorAll('[data-slide]');
                            const current = document.querySelector('.slide.active');
                            if (!current) return null;
                            
                            const title = current.querySelector('.slide-title')?.textContent || 'Unknown';
                            const index = Array.from(slides).indexOf(current) + 1;
                            
                            return {
                                title: title,
                                index: index,
                                total: slides.length
                            };
                        }
                    """)
                    
                    if current_info:
                        logger.info(f"📍 Slide {current_info['index']}/{current_info['total']}: {current_info['title']}")
                except Exception as e:
                    logger.warning(f"Could not get slide info: {e}")
                
                # Wait for narration to be read (pause time)
                await page.wait_for_timeout(slide_pause_ms)
                
                # Check if we're on the last slide
                try:
                    is_last = await page.evaluate("""
                        () => {
                            const slides = document.querySelectorAll('[data-slide]');
                            const current = document.querySelector('.slide.active');
                            return Array.from(slides).indexOf(current) === slides.length - 1;
                        }
                    """)
                    
                    if is_last:
                        logger.info("✅ Reached last slide")
                        await page.wait_for_timeout(2000)  # Final pause
                        break
                except Exception as e:
                    logger.warning(f"Could not determine if last slide: {e}")
                    break
                
                # Advance to next slide
                if auto_advance:
                    try:
                        await page.keyboard.press("ArrowRight")
                        await page.wait_for_timeout(500)  # Transition animation
                    except Exception as e:
                        logger.warning(f"Could not advance slide: {e}")
                        break
            
            logger.info(f"✅ Recorded {slide_count} slides")
            
            # Close and save video
            await context.close()
            await browser.close()
            
            if self.record_video:
                # Playwright saves video to output_dir
                saved_videos = list(self.output_dir.glob("*.webm"))
                if saved_videos:
                    latest_video = max(saved_videos, key=lambda p: p.stat().st_mtime)
                    logger.info(f"✅ Video saved: {latest_video}")
                    return latest_video
        
        return None


async def run_recording(args: dict):
    """Run the recording with provided arguments"""
    recorder = DemoRecorder(
        html_file=args.get("html_file", "demo-slides.html"),
        output_dir=args.get("output_dir", "recordings"),
        headless=args.get("headless", False),
        record_video=args.get("record_video", True),
        slow_motion=args.get("slow_motion", 500)
    )
    
    video_file = await recorder.record(
        auto_advance=args.get("auto_advance", True),
        slide_pause_ms=args.get("slide_pause_ms", 2000),
        max_slides=args.get("max_slides", None)
    )
    
    if video_file:
        logger.info(f"\n{'='*60}")
        logger.info(f"🎉 Recording complete!")
        logger.info(f"{'='*60}")
        logger.info(f"Video: {video_file.relative_to(Path.cwd())}")
        logger.info(f"Size: {video_file.stat().st_size / (1024*1024):.1f} MB")
        logger.info(f"\nYou can now:")
        logger.info(f"  1. Edit with video editor (trim, add narration, etc.)")
        logger.info(f"  2. Upload to YouTube/Vimeo")
        logger.info(f"  3. Share with stakeholders")
        logger.info(f"{'='*60}")
    else:
        logger.error("❌ Recording failed - no video file generated")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Automated Demo Recorder - Records presentation slides with Playwright"
    )
    
    parser.add_argument(
        "--html",
        default="demo-slides.html",
        help="Path to HTML presentation file (default: demo-slides.html)"
    )
    
    parser.add_argument(
        "--output",
        default="recordings",
        help="Output directory for recordings (default: recordings)"
    )
    
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run browser in headless mode (no GUI)"
    )
    
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Disable video recording (testing mode)"
    )
    
    parser.add_argument(
        "--slow-mo",
        type=int,
        default=500,
        help="Slow motion speed in milliseconds (default: 500)"
    )
    
    parser.add_argument(
        "--pause",
        type=int,
        default=2000,
        help="Pause between slides in milliseconds (default: 2000)"
    )
    
    parser.add_argument(
        "--max-slides",
        type=int,
        default=None,
        help="Maximum slides to record (default: all)"
    )
    
    parser.add_argument(
        "--no-auto-advance",
        action="store_true",
        help="Disable automatic slide advancement (manual only)"
    )
    
    args = parser.parse_args()
    
    # Check if presentation file exists
    if not Path(args.html).exists():
        logger.error(f"❌ Presentation file not found: {args.html}")
        logger.info(f"💡 First run: python AUTO_DEMO_SLIDES_EXPANDED.py")
        sys.exit(1)
    
    # Prepare recording arguments
    recording_args = {
        "html_file": args.html,
        "output_dir": args.output,
        "headless": args.headless,
        "record_video": not args.no_video,
        "slow_motion": args.slow_mo,
        "auto_advance": not args.no_auto_advance,
        "slide_pause_ms": args.pause,
        "max_slides": args.max_slides
    }
    
    print(f"""
╔═══════════════════════════════════════════════════════════╗
║         🎬 AUTOMATED DEMO RECORDER - Playwright            ║
╚═══════════════════════════════════════════════════════════╝

Configuration:
  📄 Presentation: {args.html}
  📁 Output: {args.output}/
  🎥 Video Recording: {not args.no_video}
  ⏱️  Pause per slide: {args.pause}ms
  🐢 Slow motion: {args.slow_mo}ms
  🔄 Auto-advance: {not args.no_auto_advance}
  
Press Ctrl+C to stop recording anytime.
    """)
    
    try:
        asyncio.run(run_recording(recording_args))
    except KeyboardInterrupt:
        logger.info("\n⏹️  Recording stopped by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"❌ Recording failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
