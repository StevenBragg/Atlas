#!/usr/bin/env python3
"""
Automate Salad Cloud project creation using Playwright
"""

import asyncio
from playwright.async_api import async_playwright

# Your credentials
SALAD_API_KEY = "salad_cloud_user_De2gyyIRSnQP5DQV5T5B2gA0ArR6ZAIuIxkIxJ4rsNR0zVAId"
ORG_NAME = "stevenbragg"
PROJECT_NAME = "atlas"

async def create_salad_project():
    async with async_playwright() as p:
        # Launch browser
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context()
        page = await context.new_page()
        
        try:
            print("Navigating to Salad Cloud portal...")
            await page.goto("https://portal.salad.com/")
            
            # Wait for page to load
            await page.wait_for_load_state("networkidle")
            print(f"Page title: {await page.title()}")
            
            # Check if already logged in (look for dashboard elements)
            try:
                await page.wait_for_selector("text=Dashboard", timeout=5000)
                print("Already logged in!")
            except:
                print("Need to log in...")
                # The API key should work for API access, but web UI may need different auth
                print("Note: Web UI may require separate login")
                
                # Try to use API key for authentication if there's a field
                try:
                    api_key_input = await page.wait_for_selector("input[placeholder*='API' i], input[name*='key' i]", timeout=3000)
                    if api_key_input:
                        await api_key_input.fill(SALAD_API_KEY)
                        await page.click("button[type='submit']")
                        await page.wait_for_load_state("networkidle")
                except:
                    print("No API key field found on login page")
            
            # Take screenshot to see current state
            await page.screenshot(path="/tmp/salad_portal.png")
            print("Screenshot saved to /tmp/salad_portal.png")
            
            # Try to find and click "New Project" button
            try:
                new_project_btn = await page.wait_for_selector("text=New Project, button:has-text('New Project'), [data-testid*='new-project'], a:has-text('New Project')", timeout=5000)
                if new_project_btn:
                    print("Found New Project button, clicking...")
                    await new_project_btn.click()
                    
                    # Fill in project name
                    await page.fill("input[name='name'], input[placeholder*='name' i]", PROJECT_NAME)
                    
                    # Fill in description (optional)
                    try:
                        await page.fill("input[name='description'], textarea", "Atlas AI self-organizing learning system")
                    except:
                        pass
                    
                    # Click create
                    await page.click("button[type='submit'], button:has-text('Create')")
                    
                    # Wait for creation
                    await page.wait_for_load_state("networkidle")
                    
                    print(f"✅ Project '{PROJECT_NAME}' created successfully!")
                    
                    # Take final screenshot
                    await page.screenshot(path="/tmp/salad_project_created.png")
                    
                else:
                    print("❌ Could not find 'New Project' button")
                    
            except Exception as e:
                print(f"❌ Error creating project: {e}")
                await page.screenshot(path="/tmp/salad_error.png")
                print("Error screenshot saved to /tmp/salad_error.png")
                
        except Exception as e:
            print(f"❌ Browser automation error: {e}")
            
        finally:
            await browser.close()

if __name__ == "__main__":
    asyncio.run(create_salad_project())
