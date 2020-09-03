import gzip
from multiprocessing import Pool

from tqdm import tqdm
import json
import fire
from pathlib import Path


def process_file(filename, out_folder, do_lower_case):
    outfile = Path(out_folder) / (Path(filename).name.split(".")[0] + ".txt")
    print(f"Processing {filename}, save to {outfile}, lowercasing = {do_lower_case}")

    with gzip.open(filename, 'rt') as f:
        with open(outfile, 'wt') as out:
            for i, line in tqdm(enumerate(f)):
                doc = json.loads(line)
                if doc["length"] > 256:
                    content = doc["raw_content"]
                    if do_lower_case:
                        content = content.lower()
                    out.write(content + "\n\n")


def main(inp_dir, out_dir, do_lower_case=False, num_workers=1):
    file_paths = list(Path(inp_dir).glob("*json.gz"))
    writer_workers = Pool(min(num_workers, len(file_paths)))
    arguments = [(fl, out_dir, do_lower_case) for fl in file_paths]
    writer_workers.starmap(process_file, arguments)


if __name__ == "__main__":
    fire.Fire(main)
