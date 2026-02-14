"""
Day 36: Markdown to HTML Converter
Convert markdown files to styled HTML pages
"""

import re
import sys
from pathlib import Path
from datetime import datetime


class MarkdownConverter:
    """Convert Markdown to HTML"""
    
    def __init__(self, style='default'):
        self.style = style
    
    def convert(self, markdown_text):
        """Convert markdown to HTML"""
        
        html = markdown_text
        
        # Headers (must come before bold/italic)
        html = re.sub(r'^######\s+(.+)$', r'<h6>\1</h6>', html, flags=re.MULTILINE)
        html = re.sub(r'^#####\s+(.+)$', r'<h5>\1</h5>', html, flags=re.MULTILINE)
        html = re.sub(r'^####\s+(.+)$', r'<h4>\1</h4>', html, flags=re.MULTILINE)
        html = re.sub(r'^###\s+(.+)$', r'<h3>\1</h3>', html, flags=re.MULTILINE)
        html = re.sub(r'^##\s+(.+)$', r'<h2>\1</h2>', html, flags=re.MULTILINE)
        html = re.sub(r'^#\s+(.+)$', r'<h1>\1</h1>', html, flags=re.MULTILINE)
        
        # Bold and Italic
        html = re.sub(r'\*\*\*(.+?)\*\*\*', r'<strong><em>\1</em></strong>', html)
        html = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', html)
        html = re.sub(r'\*(.+?)\*', r'<em>\1</em>', html)
        html = re.sub(r'__(.+?)__', r'<strong>\1</strong>', html)
        html = re.sub(r'_(.+?)_', r'<em>\1</em>', html)
        
        # Inline code
        html = re.sub(r'`(.+?)`', r'<code>\1</code>', html)
        
        # Links
        html = re.sub(r'\[(.+?)\]\((.+?)\)', r'<a href="\2">\1</a>', html)
        
        # Images
        html = re.sub(r'!\[(.+?)\]\((.+?)\)', r'<img src="\2" alt="\1">', html)
        
        # Code blocks
        html = re.sub(
            r'```(\w+)?\n(.*?)```',
            lambda m: f'<pre><code class="language-{m.group(1) or "text"}">{m.group(2)}</code></pre>',
            html,
            flags=re.DOTALL
        )
        
        # Horizontal rules
        html = re.sub(r'^---$', '<hr>', html, flags=re.MULTILINE)
        html = re.sub(r'^\*\*\*$', '<hr>', html, flags=re.MULTILINE)
        
        # Lists
        html = self._convert_lists(html)
        
        # Blockquotes
        html = re.sub(r'^>\s+(.+)$', r'<blockquote>\1</blockquote>', html, flags=re.MULTILINE)
        
        # Paragraphs (lines not already in tags)
        lines = html.split('\n')
        processed = []
        
        for line in lines:
            line = line.strip()
            if line and not re.match(r'^<[^>]+>.*</[^>]+>$', line) and not line.startswith('<'):
                processed.append(f'<p>{line}</p>')
            else:
                processed.append(line)
        
        return '\n'.join(processed)
    
    def _convert_lists(self, text):
        """Convert markdown lists to HTML"""
        
        lines = text.split('\n')
        result = []
        in_ul = False
        in_ol = False
        
        for line in lines:
            stripped = line.strip()
            
            # Unordered list
            if stripped.startswith('- ') or stripped.startswith('* '):
                if not in_ul:
                    result.append('<ul>')
                    in_ul = True
                if in_ol:
                    result.append('</ol>')
                    in_ol = False
                
                item = stripped[2:]
                result.append(f'  <li>{item}</li>')
            
            # Ordered list
            elif re.match(r'^\d+\.\s+', stripped):
                if not in_ol:
                    result.append('<ol>')
                    in_ol = True
                if in_ul:
                    result.append('</ul>')
                    in_ul = False
                
                item = re.sub(r'^\d+\.\s+', '', stripped)
                result.append(f'  <li>{item}</li>')
            
            else:
                if in_ul:
                    result.append('</ul>')
                    in_ul = False
                if in_ol:
                    result.append('</ol>')
                    in_ol = False
                
                result.append(line)
        
        if in_ul:
            result.append('</ul>')
        if in_ol:
            result.append('</ol>')
        
        return '\n'.join(result)
    
    def get_css(self):
        """Get CSS styles"""
        
        styles = {
            'default': """
body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
    line-height: 1.6;
    max-width: 800px;
    margin: 0 auto;
    padding: 20px;
    color: #333;
    background: #fff;
}

h1, h2, h3, h4, h5, h6 {
    margin-top: 24px;
    margin-bottom: 16px;
    font-weight: 600;
    line-height: 1.25;
}

h1 { font-size: 2em; border-bottom: 1px solid #eee; padding-bottom: 8px; }
h2 { font-size: 1.5em; border-bottom: 1px solid #eee; padding-bottom: 8px; }
h3 { font-size: 1.25em; }

code {
    background: #f6f8fa;
    padding: 2px 6px;
    border-radius: 3px;
    font-family: 'Monaco', 'Courier New', monospace;
    font-size: 0.9em;
}

pre {
    background: #f6f8fa;
    padding: 16px;
    border-radius: 6px;
    overflow-x: auto;
}

pre code {
    background: none;
    padding: 0;
}

blockquote {
    border-left: 4px solid #ddd;
    padding-left: 16px;
    margin-left: 0;
    color: #666;
}

a {
    color: #0366d6;
    text-decoration: none;
}

a:hover {
    text-decoration: underline;
}

img {
    max-width: 100%;
    height: auto;
}

hr {
    border: 0;
    border-top: 1px solid #eee;
    margin: 24px 0;
}

ul, ol {
    padding-left: 30px;
}

li {
    margin: 4px 0;
}
""",
            'dark': """
body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    line-height: 1.6;
    max-width: 800px;
    margin: 0 auto;
    padding: 20px;
    color: #c9d1d9;
    background: #0d1117;
}

h1, h2, h3 { color: #58a6ff; }
h1 { border-bottom: 1px solid #21262d; }
h2 { border-bottom: 1px solid #21262d; }

code {
    background: #161b22;
    padding: 2px 6px;
    border-radius: 3px;
    color: #f85149;
}

pre {
    background: #161b22;
    padding: 16px;
    border-radius: 6px;
    border: 1px solid #30363d;
}

blockquote {
    border-left: 4px solid #30363d;
    color: #8b949e;
}

a { color: #58a6ff; }
hr { border-top: 1px solid #21262d; }
""",
            'minimal': """
body {
    font-family: Georgia, serif;
    line-height: 1.8;
    max-width: 700px;
    margin: 40px auto;
    padding: 20px;
    color: #222;
}

h1, h2, h3 { font-weight: normal; }
code { background: #eee; padding: 2px 4px; }
pre { background: #eee; padding: 12px; }
a { color: #000; border-bottom: 1px solid #000; }
"""
        }
        
        return styles.get(self.style, styles['default'])
    
    def create_html_page(self, content, title='Document'):
        """Create complete HTML page"""
        
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
{self.get_css()}
    </style>
</head>
<body>
{content}
<footer style="margin-top: 40px; padding-top: 20px; border-top: 1px solid #eee; color: #666; font-size: 0.9em;">
    <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
</footer>
</body>
</html>"""


def convert_file(input_file, output_file=None, style='default'):
    """Convert markdown file to HTML"""
    
    input_path = Path(input_file)
    
    if not input_path.exists():
        print(f"❌ File not found: {input_file}")
        return
    
    # Read markdown
    with open(input_path, 'r', encoding='utf-8') as f:
        markdown = f.read()
    
    # Convert
    converter = MarkdownConverter(style)
    html_content = converter.convert(markdown)
    
    # Get title from first heading or filename
    title_match = re.search(r'^#\s+(.+)$', markdown, re.MULTILINE)
    title = title_match.group(1) if title_match else input_path.stem
    
    # Create full HTML page
    html_page = converter.create_html_page(html_content, title)
    
    # Output file
    if output_file is None:
        output_file = input_path.with_suffix('.html')
    
    # Write HTML
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_page)
    
    print(f"✅ Converted: {input_file}")
    print(f"📄 Output: {output_file}")
    print(f"🎨 Style: {style}")


def main():
    """CLI"""
    
    print("\n" + "="*60)
    print("📝 MARKDOWN TO HTML CONVERTER")
    print("="*60)
    
    if len(sys.argv) < 2 or '--help' in sys.argv:
        print("""
Usage:
  python3 md2html.py <input.md> [output.html] [--style=STYLE]

Styles:
  default  - GitHub-like (default)
  dark     - Dark theme
  minimal  - Minimalist serif

Examples:
  python3 md2html.py README.md
  python3 md2html.py notes.md blog.html
  python3 md2html.py doc.md --style=dark
""")
        return
    
    input_file = sys.argv[1]
    output_file = None
    style = 'default'
    
    # Parse arguments
    for arg in sys.argv[2:]:
        if arg.startswith('--style='):
            style = arg.split('=')[1]
        elif not output_file and not arg.startswith('--'):
            output_file = arg
    
    convert_file(input_file, output_file, style)
    print()


if __name__ == "__main__":
    main()
