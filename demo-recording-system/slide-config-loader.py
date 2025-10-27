#!/usr/bin/env python3
"""
Slide Configuration Loader - Modular Design
Loads slide definitions from YAML and generates HTML presentation
Keeps content separate from code for easy management

Architecture:
  - slides/*.yaml: Modular content files (12 sections)
  - presentation-template.html: Reusable HTML/CSS/JS template
  - This file: Core logic only (no embedded HTML)

Features:
  - Split-screen layout (horizontal on desktop, vertical on mobile)
  - Large narration text formatted as bullet points
  - Responsive design with media queries
  - Self-contained HTML (no external dependencies)
"""

import yaml
from pathlib import Path
from typing import List
from dataclasses import dataclass


@dataclass
class Slide:
    """Represents a single slide"""
    id: str
    title: str
    narration: str
    duration: int = 25
    content_type: str = "text"
    content: str = ""
    section: str = ""  # Added to track which section this slide belongs to
    
    def to_html(self) -> str:
        """Convert slide to HTML with split-screen layout"""
        content_html = self._format_content()
        narration_html = self._format_narration()
        
        return f"""
        <div class="slide" data-slide="{self.id}" data-duration="{self.duration}" data-section="{self.section}">
            <div class="slide-container">
                <div class="slide-left">
                    <div class="slide-header">
                        <h1 class="slide-title">{self.title}</h1>
                        <div class="slide-timer" data-duration="{self.duration}">00:{self.duration:02d}</div>
                    </div>
                    
                    <div class="narration-section">
                        {narration_html}
                    </div>
                </div>
                
                <div class="slide-divider"></div>
                
                <div class="slide-right">
                    {content_html}
                </div>
            </div>
            
            <div class="slide-footer">
                <p class="slide-id">{self.id}</p>
            </div>
        </div>
        """
    
    def _format_narration(self) -> str:
        """Format narration as large bullet points"""
        if not self.narration:
            return ""
        
        lines = [line.strip() for line in self.narration.split('\n') if line.strip()]
        
        if len(lines) == 1:
            return f'<ul class="narration"><li>{lines[0]}</li></ul>'
        
        items = '\n'.join(f'<li>{line}</li>' for line in lines)
        return f'<ul class="narration">\n{items}\n</ul>'
    
    def _format_content(self) -> str:
        """Format slide content based on type"""
        if not self.content:
            return '<div class="content-placeholder"></div>'
        
        if self.content_type == "code":
            # Preserve whitespace for ASCII art diagrams and code
            escaped = self.content.strip()
            return f'<pre class="code-block"><code>{escaped}</code></pre>'
        
        elif self.content_type == "text":
            # Preserve whitespace for tree diagrams and ASCII art
            escaped = self.content.strip()
            return f'<pre class="text-content">{escaped}</pre>'
        
        elif self.content_type == "diagram":
            # ASCII diagrams - preserve all whitespace
            escaped = self.content.strip()
            return f'<pre class="code-block"><code>{escaped}</code></pre>'
        
        else:
            # HTML content as-is
            return f'<div class="content">{self.content}</div>'


@dataclass
class Section:
    """Represents a section of slides"""
    name: str
    slides: List[Slide]
    
    def total_duration(self) -> int:
        """Calculate total duration for section"""
        return sum(slide.duration for slide in self.slides)


class SlidesConfigLoader:
    """Loads and manages slide configuration from YAML"""
    
    def __init__(self, config_dir: str = "slides", legacy_config: str = "slides-config.yaml"):
        """
        Initialize loader
        
        Args:
            config_dir: Directory containing modular YAML files (01-*.yaml, 02-*.yaml, etc.)
            legacy_config: Fallback to single monolithic YAML file
        """
        self.config_dir = Path(config_dir)
        self.legacy_config = Path(legacy_config)
        self.template_file = Path("presentation-template.html")
        self.config = None
        self.sections: List[Section] = []
        self.slides: List[Slide] = []
        
    def load(self) -> bool:
        """Load configuration from modular YAML files or legacy single file"""
        if self.config_dir.exists():
            return self._load_modular()
        elif self.legacy_config.exists():
            return self._load_legacy()
        else:
            print(f"❌ Config not found: {self.config_dir} or {self.legacy_config}")
            return False
    
    def _load_modular(self) -> bool:
        """Load configuration from modular YAML files in slides/ directory"""
        yaml_files = sorted(self.config_dir.glob("*.yaml"))
        
        if not yaml_files:
            print(f"❌ No YAML files found in {self.config_dir}")
            return False
        
        try:
            print(f"✅ Loading modular config from {self.config_dir}/")
            all_sections = []
            
            for yaml_file in yaml_files:
                with open(yaml_file, 'r') as f:
                    file_config = yaml.safe_load(f)
                
                if file_config and 'sections' in file_config:
                    all_sections.extend(file_config['sections'])
            
            self.config = {'sections': all_sections}
            self._parse_sections()
            return True
            
        except Exception as e:
            print(f"❌ Error loading modular config: {e}")
            return False
    
    def _load_legacy(self) -> bool:
        """Load configuration from single monolithic YAML file (legacy)"""
        try:
            with open(self.legacy_config, 'r') as f:
                self.config = yaml.safe_load(f)
            
            print(f"✅ Loaded config (legacy): {self.legacy_config}")
            self._parse_sections()
            return True
            
        except Exception as e:
            print(f"❌ Error loading legacy config: {e}")
            return False
    
    def _parse_sections(self):
        """Parse sections and slides from loaded config"""
        if not self.config or 'sections' not in self.config:
            print("❌ Invalid config: no sections found")
            return
        
        for section_data in self.config['sections']:
            section_name = section_data.get('name', 'Unknown')
            slides_data = section_data.get('slides', [])
            
            section_slides = []
            for slide_data in slides_data:
                slide = Slide(
                    id=slide_data.get('id', ''),
                    title=slide_data.get('title', ''),
                    narration=slide_data.get('narration', '').strip(),
                    duration=slide_data.get('duration', 25),
                    content_type=slide_data.get('content_type', 'text'),
                    content=slide_data.get('content', '').strip(),
                    section=section_name  # Add section name to slide
                )
                section_slides.append(slide)
                self.slides.append(slide)
            
            section = Section(name=section_name, slides=section_slides)
            self.sections.append(section)
            print(f"  📚 {section_name}: {len(section_slides)} slides ({section.total_duration()}s)")
    
    def get_total_duration_minutes(self) -> float:
        """Get total presentation duration in minutes"""
        total_seconds = sum(slide.duration for slide in self.slides)
        return total_seconds / 60.0
    
    def generate_html(self, output_file: str = "demo-slides.html") -> str:
        """Generate HTML presentation from loaded config"""
        html_content = self._build_html()
        
        output_path = Path(output_file)
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        print(f"\n✅ Generated: {output_path}")
        return str(output_path)
    
    def _build_html(self) -> str:
        """Build HTML by loading template and injecting slides"""
        slides_html = '\n'.join(slide.to_html() for slide in self.slides)
        
        # Load external template
        if self.template_file.exists():
            with open(self.template_file, 'r') as f:
                template = f.read()
            return template.replace('{slides_html}', slides_html)
        else:
            # Fallback if template not found
            print("⚠️  Template not found, using minimal fallback version")
            return self._build_html_minimal(slides_html)
    
    def _build_html_minimal(self, slides_html: str) -> str:
        """Minimal fallback HTML when template not found"""
        return f"""<!DOCTYPE html>
<html>
<head>
    <title>AI-Native Architecture Demo</title>
    <style>
        body {{ font-family: sans-serif; background: #1e3a8a; color: #f1f5f9; }}
        .slide {{ padding: 40px; margin: 20px; border: 1px solid #60a5fa; }}
        .slide-title {{ font-size: 2rem; color: #60a5fa; }}
        .narration {{ font-size: 1.5rem; line-height: 2; color: #e2e8f0; }}
        .code-block {{ background: #1e293b; padding: 15px; border-radius: 8px; overflow-x: auto; }}
    </style>
</head>
<body>
    <h1 style="text-align:center; color:#60a5fa;">AI-Native Architecture Presentation</h1>
    <p style="text-align:center; color:#fbbf24;">⚠️ Fallback mode: Template not loaded</p>
    {slides_html}
</body>
</html>
"""


def main():
    """Main entry point"""
    print(f"""
╔═══════════════════════════════════════════════════════════╗
║         📋 SLIDE CONFIGURATION LOADER                      ║
║         Split-Screen Responsive Layout                     ║
║         Modular YAML + External Template                   ║
╚═══════════════════════════════════════════════════════════╝
    """)
    
    loader = SlidesConfigLoader(config_dir="slides", legacy_config="slides-config.yaml")
    
    if not loader.load():
        print("❌ Failed to load configuration")
        return 1
    
    print(f"\n📊 Presentation Summary:")
    print(f"  📄 Total slides: {len(loader.slides)}")
    print(f"  ⏱️  Total duration: {loader.get_total_duration_minutes():.1f} minutes")
    print(f"  📚 Sections: {len(loader.sections)}\n")
    
    for i, section in enumerate(loader.sections, 1):
        print(f"  {i}. {section.name}")
    
    # Generate HTML
    output_file = loader.generate_html()
    
    print(f"\n{'='*60}")
    print(f"✅ Presentation ready!")
    print(f"{'='*60}")
    print(f"Open in browser: file://{Path(output_file).resolve()}")
    print(f"\nFeatures:")
    print(f"  ✨ Split-screen layout (horizontal on desktop)")
    print(f"  ✨ Vertical layout on mobile (< 1024px)")
    print(f"  ✨ Large narration as bullet points")
    print(f"  ✨ Responsive design")
    print(f"\nControls:")
    print(f"  ← → : Navigate slides")
    print(f"  SPACE : Play/Pause")
    print(f"  ESC : Stop playback")
    
    return 0


if __name__ == "__main__":
    try:
        import sys
        sys.exit(main())
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
