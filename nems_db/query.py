import subprocess


def grep_dirtree(abspath, contains, include='*.json'):
    '''
    Greps recursively through the directory tree at path looking for files that
    contain contains. By default, it looks at all JSON files, but you
    may use the optional argument 'include' to change that.

    It also removes the 'path' string at the beginning of each line.
    '''
    cmd = ['grep', '-r', '-i', '-l', '--include', include,
           contains, abspath]

    result = subprocess.run(cmd,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.DEVNULL)

    ret = result.returncode
    stdout = result.stdout.decode()

    if ret == 0 or ret == 1:
        lines = stdout.splitlines()
        if abspath[-1] == '/':
            n = len(abspath) - 1
        else:
            n = len(abspath)
        unprefixed_lines = [l[n:] for l in lines]  # Strip abspath from front
        return unprefixed_lines
    else:
        raise ValueError('A grep error occurred for {}'.format(str(cmd)))
