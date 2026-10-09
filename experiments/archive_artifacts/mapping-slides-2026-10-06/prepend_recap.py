from pathlib import Path
import re

ROOT = Path('/Users/jihoonkwon/Desktop/projects/project-pages/f4750a4b/multimodal-SAE')

def prepend_recap(page):
    source = (ROOT / '2026-09-30.html').read_text()
    sections = re.findall(r'<section\b.*?</section>', source, re.S)
    selected = []
    for number in (3, 5, 6, 7):
        section = sections[number - 1]
        section = section.replace('class="slide ', 'class="slide recap-import ')
        section = re.sub(r'id="slide-(\d+)"', r'id="recap-\1"', section)
        section = re.sub(r' aria-label="[^"]*"', '', section, count=1)
        selected.append(section)
    selected.append(Path(__file__).with_name('meeting-discussion.html').read_text())
    css = '\n'.join(re.findall(r'<style[^>]*>(.*?)</style>', source, re.S))
    css = css.replace(':root', ':scope')
    css = re.sub(r'#slide-(\d+)', r'#recap-\1', css)
    # Isolate the original deck's typography and layout from the current deck.
    css += '\n:scope{line-height:1.4} .content{height:auto} aside.speaker-notes{display:none}'
    page = page.replace('</head>', '<style id="recap-styles">@scope (.recap-import){' + css + '}</style></head>', 1)
    return page.replace('<div class="stage">', '<div class="stage">' + ''.join(selected), 1)

if __name__ == '__main__':
    target = ROOT / '2026-10-06.html'
    page = target.read_text()
    if 'id="recap-styles"' not in page:
        target.write_text(prepend_recap(page))
