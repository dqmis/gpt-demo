import glob

DATA_DIR = "./data/poetry"


def process_file(filename: str) -> str:
    with open(filename, "r") as f:
        lines = f.readlines()
    return "".join(
        [line for line in lines if (line[0].isalpha() and not line[:2].isupper())]
    )


if __name__ == "__main__":
    full_text = ""
    for file in glob.glob(f"{DATA_DIR}/*"):
        full_text += process_file(file)

    full_text_lines = full_text.splitlines()

    mean_line_len = sum([len(l) for l in full_text_lines]) / len(full_text_lines)

    full_text_lines = [l for l in full_text_lines if len(l) < mean_line_len]

    with open("eiles.txt", "w") as f:
        f.write("\n".join(full_text_lines))
