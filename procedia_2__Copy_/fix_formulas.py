import re

# Read the LaTeX file
with open('PROCS_ ICMLDE 2025..tex', 'r', encoding='utf-8') as f:
    content = f.read()

# Fix all \bm{\textbf{...} patterns by removing the \bm{ wrapper
# This pattern handles nested braces correctly
pattern = r'\\bm\{\\textbf\{([^}]+(?:\{[^}]*\}[^}]*)*)\}\}'
content = re.sub(pattern, r'\\textbf{\1}', content)

# Fix patterns like \bm{\textbf{...}} where there are double closing braces
pattern2 = r'\\bm\{\\textbf\{([^}]+(?:\{[^}]*\}[^}]*)*)\}\}'
content = re.sub(pattern2, r'\\textbf{\1}', content)

# Fix any remaining \bm{\textbf patterns without proper closing
pattern3 = r'\\bm\{\\textbf\{([^}]+)\}\}'
content = re.sub(pattern3, r'\\textbf{\1}', content)

# Fix simple mathematical expressions wrapped in \bm{}
pattern4 = r'\\bm\{([^{}]+)\}'
content = re.sub(pattern4, r'\1', content)

# Write back to file
with open('PROCS_ ICMLDE 2025..tex', 'w', encoding='utf-8') as f:
    f.write(content)

print("Formula patterns fixed!")
