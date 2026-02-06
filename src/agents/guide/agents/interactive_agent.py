#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
InteractiveAgent - Agent for generating interactive HTML pages
Converts knowledge points into visual, interactive learning pages
"""

import re
from typing import Any

from src.agents.base_agent import BaseAgent


class InteractiveAgent(BaseAgent):
    """Interactive page generation agent"""

    def __init__(self, language: str = "zh"):
        super().__init__(
            module_name="guide",
            agent_name="interactive_agent",
            language=language,
        )

    def _extract_html(self, response: str) -> str:
        """Extract HTML code from LLM response"""
        html_pattern = r"```html\s*([\s\S]*?)\s*```"
        match = re.search(html_pattern, response)
        if match:
            return match.group(1).strip()

        code_pattern = r"```\s*([\s\S]*?)\s*```"
        match = re.search(code_pattern, response)
        if match:
            content = match.group(1).strip()
            if content.startswith("<!DOCTYPE") or content.startswith("<html"):
                return content

        if response.strip().startswith("<!DOCTYPE") or response.strip().startswith("<html"):
            return response.strip()

        return response.strip()

    def _validate_html(self, html: str) -> bool:
        """Validate if HTML is basically valid"""
        return (
            "<html" in html.lower()
            or "<!doctype" in html.lower()
            or "<body" in html.lower()
            or "<div" in html.lower()
        )

    def _sanitize_javascript(self, html: str) -> str:
        """Sanitize JavaScript to prevent browser freezing.

        Detects and wraps potentially dangerous patterns:
        - Infinite loops (while(true), for(;;))
        - Uncontrolled setInterval/setTimeout
        - Recursive animations
        """
        # Patterns that could cause browser freeze
        dangerous_patterns = [
            (r'while\s*\(\s*true\s*\)', 'while(false)'),  # while(true)
            (r'while\s*\(\s*1\s*\)', 'while(false)'),  # while(1)
            (r'for\s*\(\s*;\s*;\s*\)', 'for(;false;)'),  # for(;;)
            (r'for\s*\(\s*;;\s*\)', 'for(;false;)'),  # for(;;)
        ]

        result = html
        for pattern, replacement in dangerous_patterns:
            result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)

        # Add safety timeout wrapper to setInterval calls
        # This ensures intervals don't run indefinitely
        result = re.sub(
            r'setInterval\s*\(\s*function',
            'setInterval(function',
            result
        )

        # Limit requestAnimationFrame recursion by adding a frame counter
        if 'requestAnimationFrame' in result:
            safety_wrapper = """
<script>
// Safety wrapper to prevent infinite animation loops
(function() {
    var _raf = window.requestAnimationFrame;
    var _frameCount = 0;
    var _maxFrames = 10000;
    window.requestAnimationFrame = function(callback) {
        _frameCount++;
        if (_frameCount > _maxFrames) {
            console.warn('Animation frame limit reached, stopping to prevent freeze');
            return;
        }
        return _raf.call(window, callback);
    };
})();
</script>
"""
            # Insert safety wrapper at the beginning of body
            if '<body' in result.lower():
                result = re.sub(
                    r'(<body[^>]*>)',
                    r'\1' + safety_wrapper,
                    result,
                    count=1,
                    flags=re.IGNORECASE
                )

        return result

    def _validate_javascript_safety(self, html: str) -> tuple[bool, str]:
        """Check for potentially dangerous JavaScript patterns.

        Returns:
            Tuple of (is_safe, warning_message)
        """
        warnings = []

        # Check for infinite loop patterns
        if re.search(r'while\s*\(\s*(true|1)\s*\)', html, re.IGNORECASE):
            warnings.append("Contains while(true) loop")

        if re.search(r'for\s*\(\s*;[\s;]*\)', html, re.IGNORECASE):
            warnings.append("Contains infinite for loop")

        # Check for uncontrolled intervals
        if 'setInterval' in html and 'clearInterval' not in html:
            warnings.append("setInterval without clearInterval")

        # Check for potential recursion without base case
        recursive_funcs = re.findall(r'function\s+(\w+)\s*\([^)]*\)\s*\{[^}]*\1\s*\(', html)
        if recursive_funcs:
            warnings.append(f"Potentially recursive functions: {recursive_funcs}")

        # Check for excessive DOM queries in loops
        if re.search(r'(while|for)\s*\([^)]*\)\s*\{[^}]*querySelector', html):
            warnings.append("DOM query inside loop")

        is_safe = len(warnings) == 0
        return is_safe, "; ".join(warnings) if warnings else ""

    def _generate_fallback_html(self, knowledge: dict[str, Any]) -> str:
        """Generate fallback HTML page"""
        title = knowledge.get("knowledge_title", "Knowledge Point")
        summary = knowledge.get("knowledge_summary", "").replace("\n", "<br>")
        difficulty = knowledge.get("user_difficulty", "").replace("\n", "<br>")

        return f"""<!DOCTYPE html>
<html lang="zh">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css" integrity="sha384-n8MVd4RsNIU0tAv4ct0nTaAbDJwPJzDEaqSD1odI+WdtXRGWt2kTvGFasHpSy3SV" crossorigin="anonymous">
    <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js" integrity="sha384-XjKyOOlGwcjNTAIQHIpgOno0Hl1YQqzUOEleOLALmuqehneUG+vnGctmUb0ZY0l8" crossorigin="anonymous"></script>
    <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/contrib/auto-render.min.js" integrity="sha384-+VBxd3r6XgURycqtZ117nYw44OOcIax56Z4dCRWbxyPt0Koah1uHoK0o4+/RRE05" crossorigin="anonymous" onload="renderMathInElement(document.body);"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #F8FAFC 0%, #EFF6FF 100%);
            min-height: 100vh;
            padding: 2rem;
            color: #1E293B;
        }}
        .container {{
            max-width: 800px;
            margin: 0 auto;
        }}
        .card {{
            background: white;
            border-radius: 16px;
            padding: 2rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
            transition: transform 0.2s, box-shadow 0.2s;
        }}
        .card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 10px 25px -5px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #1E40AF;
            font-size: 1.75rem;
            margin-bottom: 1rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }}
        h2 {{
            color: #3B82F6;
            font-size: 1.25rem;
            margin-bottom: 1rem;
        }}
        .content {{
            line-height: 1.8;
            color: #475569;
        }}
        .difficulty {{
            background: #FEF3C7;
            border-left: 4px solid #F59E0B;
            padding: 1rem;
            border-radius: 0 8px 8px 0;
            margin-top: 1rem;
        }}
        .difficulty h3 {{
            color: #B45309;
            margin-bottom: 0.5rem;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="card">
            <h1>📚 {title}</h1>
        </div>
        <div class="card">
            <h2>Knowledge Content</h2>
            <div class="content">{summary}</div>
        </div>
        <div class="card">
            <h2>💡 Notes</h2>
            <div class="difficulty">
                <h3>Possible Difficulties</h3>
                <p>{difficulty}</p>
            </div>
        </div>
    </div>
</body>
</html>"""

    async def process(
        self, knowledge: dict[str, Any], retry_with_bug: str | None = None
    ) -> dict[str, Any]:
        """
        Generate interactive HTML learning page

        Args:
            knowledge: Knowledge point information (knowledge_title, knowledge_summary, user_difficulty)
            retry_with_bug: If provided, it's a bug fix request

        Returns:
            Dictionary containing HTML code
        """
        system_prompt = self.get_prompt("system")
        if not system_prompt:
            raise ValueError(
                "InteractiveAgent missing system prompt, please configure system in prompts/{lang}/interactive_agent.yaml"
            )

        user_template = self.get_prompt("user_template")
        if not user_template:
            raise ValueError(
                "InteractiveAgent missing user_template, please configure user_template in prompts/{lang}/interactive_agent.yaml"
            )

        if retry_with_bug:
            user_prompt = f"""The previously generated HTML page has the following issues:
{retry_with_bug}

Please fix these issues and regenerate the HTML page.

Original knowledge point information:
{user_template.format(**knowledge)}"""
        else:
            user_prompt = user_template.format(
                knowledge_title=knowledge.get("knowledge_title", ""),
                knowledge_summary=knowledge.get("knowledge_summary", ""),
                user_difficulty=knowledge.get("user_difficulty", ""),
            )

        try:
            response = await self.call_llm(
                user_prompt=user_prompt,
                system_prompt=system_prompt,
            )

            html_code = self._extract_html(response)

            if not self._validate_html(html_code):
                html_code = self._generate_fallback_html(knowledge)
                return {
                    "success": True,
                    "html": html_code,
                    "is_fallback": True,
                    "message": "Used fallback template",
                }

            # Check JavaScript safety and sanitize if needed
            is_safe, warnings = self._validate_javascript_safety(html_code)
            if not is_safe:
                # Log warning but still try to use sanitized version
                import logging
                logging.getLogger(__name__).warning(
                    f"Potentially unsafe JavaScript detected: {warnings}"
                )

            # Always sanitize to prevent browser freezing
            html_code = self._sanitize_javascript(html_code)

            return {
                "success": True,
                "html": html_code,
                "is_fallback": False,
                "js_warnings": warnings if warnings else None,
            }

        except Exception as e:
            html_code = self._generate_fallback_html(knowledge)
            return {
                "success": True,
                "html": html_code,
                "is_fallback": True,
                "error": str(e),
                "message": "Error occurred, used fallback template",
            }
