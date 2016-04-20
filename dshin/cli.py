import os

def yes_or_die(prompt, suffix='Continue? [y/n] {default}', default='y'):
    prompt = ' '.join([prompt, suffix.format(default='(default: {})'.format(default))])
    while True:
        resp = input(prompt)
        if resp == '':
            resp = default

        print('Input: {}'.format(resp), flush=True)

        if resp.lower() in ['n', 'no']:
            exit(0)
        elif resp.lower() in ['y', 'yes']:
            break

def confirm_or_die_if_exists(filepath, prompt='{path} exists.', default='y'):
    filepath = os.path.expanduser(filepath)
    if os.path.isfile(filepath):
        yes_or_die(prompt.format(path=filepath), default=default)

