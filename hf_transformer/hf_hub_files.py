from huggingface_hub import hf_hub_download, snapshot_download
import threading

def download_file(repo, folder, file):
    cache_dir = "data/hf"
    dir = hf_hub_download(
                    repo_id=repo, 
                    filename=file,
                    subfolder=folder,
                    repo_type="dataset",
                    revision="main",
                    cache_dir=cache_dir
                )
    return dir

def get_tdmpc2_mt30():
    usr = "nicklashansen"
    repo = "tdmpc2"
    tdmpc_repo = usr + "/" + repo
    tdmpc_folder = "mt30"
    num_chunks = 4
    for i in range(num_chunks):
        filename = "chunk_" + str(i) + ".pt"
        cache_dir = download_file(tdmpc_repo, tdmpc_folder, filename)
    return cache_dir[:-11]

def get_tdmpc2():
    snapshot_download(repo_id="nicklashansen/tdmpc2", 
                    repo_type="dataset",
                    revision="main",
                    cache_dir="data/hf",
                    )


if __name__ == "__main__":
    get_tdmpc2_mt30()