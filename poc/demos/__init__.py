"""
Phase 5: Chapter Demos - Master Runner
=======================================

This script provides a unified entry point for all chapter demonstrations.
Each demo showcases the architectural concepts from the AI-Native Application
Architecture whitepaper.

Available Demos:
- Chapter 5: MCP-Enabled Microservices
- Chapter 6: The Death of Traditional UI
- Chapter 7: Security in AI-Native Architecture
- Chapter 8: Context Management for Multi-Turn Conversations
- Chapter 10: Testing and Validation (Coming soon)
- Chapter 11: Training and Optimization (Coming soon)
"""

import asyncio
import sys
from typing import Callable, Dict, Optional


class DemoRunner:
    """Master demo runner for all chapter demonstrations"""

    def __init__(self):
        self.demos: Dict[str, Dict[str, any]] = {
            "5": {
                "title": "Chapter 5: MCP-Enabled Microservices",
                "file": "chapter5_mcp_microservices_demo.py",
                "class": "Chapter5Demo",
                "description": "Demonstrates how MCP enables microservices to expose tools to AI orchestrators",
            },
            "6": {
                "title": "Chapter 6: The Death of Traditional UI",
                "file": "chapter6_ui_layer_demo.py",
                "class": "Chapter6Demo",
                "description": "Shows how AI-native UIs adapt rendering based on user intent",
            },
            "7": {
                "title": "Chapter 7: Security in AI-Native Architecture",
                "file": "chapter7_security_demo.py",
                "class": "Chapter7Demo",
                "description": "Explores security mechanisms adapted for AI-orchestrated systems",
            },
            "8": {
                "title": "Chapter 8: Context Management for Multi-Turn Conversations",
                "file": "chapter8_context_demo.py",
                "class": "Chapter8Demo",
                "description": "Demonstrates stateful conversation management with intelligent context",
            },
        }

    def print_welcome(self):
        """Print welcome message"""
        print(f"\n{'█'*70}")
        print(f"{'█'*70}")
        print(f"█ {'AI-NATIVE APPLICATION ARCHITECTURE'.center(68)} █")
        print(f"█ {'Phase 5: Chapter Demonstrations'.center(68)} █")
        print(f"{'█'*70}")
        print(f"{'█'*70}\n")

    def print_menu(self):
        """Print available demos"""
        print(f"\nAvailable Demonstrations:\n")
        
        for chapter, info in sorted(self.demos.items()):
            print(f"  {chapter}. {info['title']}")
            print(f"     {info['description']}\n")
        
        print(f"  all - Run all demonstrations")
        print(f"  exit - Exit the program\n")

    def print_intro(self, chapter: str):
        """Print introduction for a demo"""
        if chapter not in self.demos:
            print(f"\n✗ Invalid chapter: {chapter}")
            return False
        
        info = self.demos[chapter]
        print(f"\n{'='*70}")
        print(f"Running: {info['title']}")
        print(f"{'='*70}")
        return True

    async def run_demo(self, chapter: str) -> bool:
        """Run a specific demo"""
        if not self.print_intro(chapter):
            return False
        
        try:
            # Import the demo module dynamically
            info = self.demos[chapter]
            module_name = info["file"].replace(".py", "")
            
            # For now, print a message about how to run the demo
            print(f"\nTo run this demo, use:")
            print(f"  python demos/{info['file']}")
            print(f"\nDemo file location:")
            print(f"  /Users/srinivas/source/poc/ai-native-application-architecture-whitepaper/poc/demos/{info['file']}")
            print(f"\nThe {info['title']} demo provides:")
            print(f"  • Interactive scenarios")
            print(f"  • Real architecture examples")
            print(f"  • Use cases from the whitepaper")
            print(f"  • Key architectural insights")
            
            return True
            
        except Exception as e:
            print(f"\n✗ Error running demo: {e}")
            import traceback
            traceback.print_exc()
            return False

    async def run_all_demos(self):
        """Run all available demos"""
        print(f"\n{'='*70}")
        print(f"Running ALL Chapter Demonstrations")
        print(f"{'='*70}")
        
        for chapter in sorted(self.demos.keys()):
            await self.run_demo(chapter)
            print()

    async def run_interactive(self):
        """Run in interactive mode"""
        self.print_welcome()
        
        while True:
            self.print_menu()
            choice = input("Select demo (5-8, all, or exit): ").strip().lower()
            
            if choice == "exit":
                print(f"\nThank you for exploring AI-Native Architecture!")
                break
            elif choice == "all":
                await self.run_all_demos()
            elif choice in self.demos:
                await self.run_demo(choice)
            else:
                print(f"\n✗ Invalid choice: {choice}")

    async def run_specific(self, chapter: str):
        """Run a specific demo"""
        if chapter not in self.demos:
            print(f"✗ Invalid chapter: {chapter}")
            print(f"Available chapters: {', '.join(sorted(self.demos.keys()))}")
            return False
        
        return await self.run_demo(chapter)


async def main():
    """Main entry point"""
    runner = DemoRunner()
    
    if len(sys.argv) > 1:
        # Run specific chapter
        chapter = sys.argv[1]
        await runner.run_specific(chapter)
    else:
        # Interactive mode
        await runner.run_interactive()


if __name__ == "__main__":
    asyncio.run(main())
