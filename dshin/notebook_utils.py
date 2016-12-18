import textwrap
from IPython.core import display


def html(html_content):
    display.display(display.HTML(textwrap.dedent(html_content.strip())))


def textarea(content='', width='100%', height='100px'):
    html(html_content='''
        <textarea style="border:1px solid #bbb; padding:0; margin:0; line-height:1; font-size:75%; font-family:monospace; width:{width}; height:{height}">{content}</textarea>
    '''.format(width=width, height=height, content=content))


def horizontal_line(color='#bbb', size=1):
    html(html_content='''
        <hr style="border-color: {color}; color: {color}; size={size}">
    '''.format(color=color, size=size))
