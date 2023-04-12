import glob

DATA_DIR = "./data/poetry"


def process_file_to_text(file_name: str) -> str:
    lines = open(file_name, "r").readlines()
    return "".join(
        [line for line in lines if (line[0].isalpha() and not line[:2].isupper())]
    )


def get_mean_line_chars(text: str) -> int:
    lines = text.splitlines()
    return sum([len(line) for line in lines]) // len(lines)


if __name__ == "__main__":
    full_text: str = ""
    for file_name in glob.glob(f"{DATA_DIR}/*.txt"):
        full_text += process_file_to_text(file_name)

    mean_line_len = get_mean_line_chars(full_text)

    parsed_full_text_lines = [
        line for line in full_text.splitlines() if len(line) <= mean_line_len
    ]

    open("eiles.txt", "w").write("\n".join(parsed_full_text_lines))
