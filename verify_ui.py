import os
import time
from playwright.sync_api import sync_playwright

ARTIFACT_DIR = r"C:\Users\katiy\.gemini\antigravity\brain\35dc14ea-5884-404c-adda-7c28e3c87208"
os.makedirs(ARTIFACT_DIR, exist_ok=True)

def run_browser_verification():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1440, "height": 900})
        page = context.new_page()

        print("1. Navigating to JobMatch Agent Web UI at http://127.0.0.1:8000...")
        page.goto("http://127.0.0.1:8000", wait_until="networkidle")
        page.wait_for_timeout(1000)

        # 1. Initial State
        ss1 = os.path.join(ARTIFACT_DIR, "01_initial_ui_state.png")
        page.screenshot(path=ss1)
        print(f"Captured initial state: {ss1}")

        # 2. Turn 1: Quick Prompt
        print("2. Executing Turn 1: Clicking quick prompt 'Remote Python/Postgres backend internships > ₹50k'...")
        page.click("button.quick-prompt:has-text('Remote Python/Postgres')")
        
        # Wait for assistant response bubble
        page.wait_for_selector("#chatMessages .text-slate-200:has-text('updated')", timeout=30000)
        page.wait_for_selector(".bg-slate-900.border-slate-800", timeout=30000)
        page.wait_for_timeout(2000)

        ss2 = os.path.join(ARTIFACT_DIR, "02_turn1_python_remote_matches.png")
        page.screenshot(path=ss2)
        print(f"Captured Turn 1 state: {ss2}")

        # 3. Turn 2: Follow-up constraint revision
        print("3. Executing Turn 2: Submitting 'Actually exclude React/frontend and focus on AI agents'...")
        page.fill("#chatInput", "Actually exclude React/frontend and focus on AI agents")
        page.click("#sendBtn")

        # Wait for loading indicator to finish and new assistant bubble
        page.wait_for_timeout(4000)

        ss3 = os.path.join(ARTIFACT_DIR, "03_turn2_exclude_react_ai_agents.png")
        page.screenshot(path=ss3)
        print(f"Captured Turn 2 state: {ss3}")

        # 4. Turn 3: Reset Memory
        print("4. Executing Reset Memory...")
        page.on("dialog", lambda dialog: dialog.accept())
        page.click("#resetBtn")
        page.wait_for_timeout(2000)

        ss4 = os.path.join(ARTIFACT_DIR, "04_memory_reset.png")
        page.screenshot(path=ss4)
        print(f"Captured Reset state: {ss4}")

        browser.close()
        print("Browser verification completed successfully!")

if __name__ == "__main__":
    run_browser_verification()
