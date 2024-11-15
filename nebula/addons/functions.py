import logging


def print_msg_box(msg, indent=1, width=None, title=None, logger_name=None):
    """
    Prints a formatted message box to the logger with an optional title.

    This function creates a visually appealing message box format for logging messages.
    It allows for indentation, custom width, and inclusion of a title. If the message is
    multiline, each line will be included in the box.

    Args:
        msg (str): The message to be displayed inside the box. Must be a string.
        indent (int, optional): The number of spaces to indent the message box. Default is 1.
        width (int, optional): The width of the message box. If not provided, it will be calculated
                               based on the longest line of the message and the title (if provided).
        title (str, optional): An optional title for the message box. Must be a string if provided.
        logger_name (str, optional): The name of the logger to use. If not provided, the root logger
                                      will be used.

    Raises:
        TypeError: If `msg` or `title` is not a string.

    Returns:
        None

    Notes:
        - The message box is bordered with decorative characters to enhance visibility in the logs.
        - If the `width` is not provided, it will automatically adjust to fit the content.
    """
    logger = logging.getLogger(logger_name) if logger_name else logging.getLogger()

    if not isinstance(msg, str):
        raise TypeError("msg parameter must be a string")  # noqa: TRY003

    lines = msg.split("\n")
    space = " " * indent
    if not width:
        width = max(map(len, lines))
        if title:
            width = max(width, len(title))
    box = f"\n╔{'═' * (width + indent * 2)}╗\n"  # upper_border
    if title:
        if not isinstance(title, str):
            raise TypeError("title parameter must be a string")  # noqa: TRY003
        box += f"║{space}{title:<{width}}{space}║\n"  # title
        box += f"║{space}{'-' * len(title):<{width}}{space}║\n"  # underscore
    box += "".join([f"║{space}{line:<{width}}{space}║\n" for line in lines])
    box += f"╚{'═' * (width + indent * 2)}╝"  # lower_border
    logger.info(box)
