def print_header(message, is_start, width=60):
    print(message.center(width, '-'))
    if not is_start:
        print('\n')  # Add extra newline for spacing
