def split_by_enclosure(s, enclosure=('{', '}')):
    open = enclosure[0]
    close = enclosure[1]
    open_count = 0
    close_count = 0
    groups = []

    for i, c in enumerate(s):
        if c == open:
            open_count += 1
            if open_count == 1:
                start = i
        elif c == close:
            close_count += 1

        if (open_count == close_count) \
                and (open_count != 0) \
                and (close_count != 0):
            end = i
            groups.append(s[start:end+1])
            open_count = 0
            close_count = 0

    return groups
